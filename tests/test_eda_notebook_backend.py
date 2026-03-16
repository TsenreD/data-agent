import pandas as pd
import pytest
from smolagents.agents import RunResult

from agents.data_collection.smolagents_backend import SmolagentsNotebookBackend


def test_build_task_includes_dataset_path_and_allowed_imports(tmp_path) -> None:
    backend = SmolagentsNotebookBackend(model=object())
    frame = pd.DataFrame({"text": ["hello world"], "label": ["pos"]})
    dataset_path = tmp_path / "data" / "unified_dataset.jsonl"
    notebook_path = tmp_path / "notebooks" / "eda.ipynb"
    inspection_summary = {"row_count": 1, "columns": ["text", "label"]}

    task = backend._build_task(frame, dataset_path, notebook_path, inspection_summary)

    assert "Notebook dataset path: ../data/unified_dataset.jsonl" in task
    assert "Observed dataset inspection summary" in task
    assert "matplotlib" in task
    assert "seaborn" in task
    assert "label distribution when available" in task
    assert "at most 10 cells total" in task


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


def test_generate_notebook_passes_empty_tool_list(monkeypatch, tmp_path) -> None:
    captured = {}

    class DummyAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["tools"] = kwargs["tools"]
            captured["instructions"] = kwargs["instructions"]
            captured["imports"] = kwargs["additional_authorized_imports"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run(self, task, max_steps: int, return_full_result: bool):
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

    monkeypatch.setattr("agents.data_collection.smolagents_backend.CodeAgent", DummyAgent)

    backend = SmolagentsNotebookBackend(model=object())
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
