import pandas as pd
import pytest
from smolagents.agents import RunResult

import agents.data_collection.smolagents_backend as smolagents_backend_module
from agents.data_collection.smolagents_backend import NOTEBOOK_BOOTSTRAP_MARKER, SmolagentsNotebookBackend


def test_build_task_includes_dataset_path_and_allowed_imports(tmp_path) -> None:
    backend = SmolagentsNotebookBackend(model=object())
    frame = pd.DataFrame({"text": ["hello world"], "label": ["pos"]})
    dataset_path = tmp_path / "data" / "unified_dataset.jsonl"
    notebook_path = tmp_path / "notebooks" / "eda.ipynb"
    inspection_summary = {"row_count": 1, "columns": ["text", "label"]}

    task = backend._build_task(frame, dataset_path, notebook_path, inspection_summary, max_steps=6, max_attempts=2)

    assert "Notebook dataset path: ../data/unified_dataset.jsonl" in task
    assert "Observed dataset inspection summary" in task
    assert "matplotlib" in task
    assert "seaborn" in task
    assert "Path`, `np`, `pd`, `plt`, `sns`, and `display`" in task
    assert "label distribution when available" in task
    assert "text-length distributions when text is available" in task
    assert "top-word analysis" not in task
    assert "at most 10 cells total" in task
    assert "You have at most 6 agent steps in this notebook-generation attempt." in task


def test_validate_notebook_rejects_unapproved_import() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["import requests\n"],
            }
        ],
    }

    with pytest.raises(ValueError, match="outside the approved EDA library set"):
        SmolagentsNotebookBackend._validate_notebook(notebook)


def test_normalize_notebook_rejects_more_than_ten_cells() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Cell\n"]} for _ in range(11)],
    }

    with pytest.raises(ValueError, match="at most 10 cells"):
        SmolagentsNotebookBackend._normalize_notebook(notebook)


def test_normalize_notebook_injects_bootstrap_into_first_code_cell() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# EDA\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "df = pd.read_json(Path('../data/unified_dataset.jsonl'), lines=True)\n",
                    "display(df.head())\n",
                    "sns.heatmap(df.isnull())\n",
                    "plt.show()\n",
                ],
            },
        ],
    }

    normalized = SmolagentsNotebookBackend._normalize_notebook(notebook)
    first_code_cell = normalized["cells"][1]
    source = "".join(first_code_cell["source"])

    assert "import pandas as pd" in source
    assert "import matplotlib.pyplot as plt" in source
    assert "import seaborn as sns" in source
    assert "import numpy as np" in source
    assert "from pathlib import Path" in source
    assert "from IPython.display import display" in source


def test_normalize_notebook_keeps_existing_bootstrap_without_duplication() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "try:\n",
                    "    from IPython.display import display\n",
                    "except ImportError:\n",
                    "    display = print\n",
                    "df = pd.read_json(Path('../data/unified_dataset.jsonl'), lines=True)\n",
                ],
            }
        ],
    }

    normalized = SmolagentsNotebookBackend._normalize_notebook(notebook)
    source = "".join(normalized["cells"][0]["source"])

    assert source.count("import pandas as pd") == 1
    assert NOTEBOOK_BOOTSTRAP_MARKER not in source


def test_generate_notebook_passes_empty_tool_list(monkeypatch, tmp_path) -> None:
    captured = {}

    class DummyAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["tools"] = kwargs["tools"]
            captured["instructions"] = kwargs["instructions"]
            captured["imports"] = kwargs["additional_authorized_imports"]
            captured["agent_max_steps"] = kwargs["max_steps"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run(self, task, max_steps: int, return_full_result: bool):
            captured["run_max_steps"] = max_steps
            notebook = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# EDA\n"]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [
                            "from pathlib import Path\n",
                            "import pandas as pd\n",
                            "df = pd.read_json(Path('../data/unified_dataset.jsonl'), lines=True)\n",
                            "df.head()\n",
                        ],
                    },
                ],
            }
            return RunResult(
                output={"notebook": notebook, "notes": ["ok"]},
                state="done",
                steps=[],
            )

    monkeypatch.setattr(smolagents_backend_module, "CodeAgent", DummyAgent)

    backend = SmolagentsNotebookBackend(model=object(), agent_config={"max_steps": 6})
    monkeypatch.setattr(
        backend,
        "inspect_dataset",
        lambda dataset_path: type(
            "Inspection",
            (),
            {
                "summary": {"row_count": 1},
                "success": True,
                "logs": [],
                "attempts": [],
                "notes": ["inspected"],
            },
        )(),
    )
    frame = pd.DataFrame({"text": ["hello"], "label": ["pos"]})
    result = backend.generate_notebook(
        frame=frame,
        dataset_path=tmp_path / "data" / "unified_dataset.jsonl",
        notebook_path=tmp_path / "notebooks" / "eda.ipynb",
    )

    assert result.success is True
    assert result.notebook is not None
    assert captured["tools"] == []
    assert "You generate a Jupyter notebook" in captured["instructions"]
    assert "seaborn" in captured["imports"]
    assert captured["agent_max_steps"] == 6
    assert captured["run_max_steps"] == 6
    assert result.notes == ["inspected", "ok"]


def test_inspect_dataset_uses_empty_tool_list(monkeypatch, tmp_path) -> None:
    backend = SmolagentsNotebookBackend(model=object())
    monkeypatch.setattr(
        backend,
        "_run_python_in_eda_sandbox",
        lambda **kwargs: '{"summary":{"row_count":1,"columns":["text"]},"notes":["saw text"]}',
    )
    result = backend.inspect_dataset(tmp_path / "data" / "unified_dataset.jsonl")

    assert result.success is True
    assert result.summary == {"row_count": 1, "columns": ["text"]}
    assert result.notes == ["saw text"]


def test_run_inspection_script_summarizes_text_lengths_without_word_content(monkeypatch, tmp_path) -> None:
    captured = {}

    backend = SmolagentsNotebookBackend(model=object())

    def fake_run_python_in_eda_sandbox(**kwargs):
        captured["script"] = kwargs["script"]
        return '{"summary":{"row_count":1},"notes":[]}'

    monkeypatch.setattr(backend, "_run_python_in_eda_sandbox", fake_run_python_in_eda_sandbox)

    backend._run_inspection_script(
        dataset_path=tmp_path / "data" / "unified_dataset.jsonl",
        mount_host_path=tmp_path,
        mount_container_path="/workspace",
    )

    script = captured["script"]

    assert "text_length_stats" in script
    assert "text_length_distribution" in script
    assert "top_words" not in script
    assert "sample_texts" not in script
