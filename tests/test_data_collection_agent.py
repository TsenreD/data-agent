import json

import pandas as pd

from agents.data_collection.data_collection_agent import DataCollectionAgent, SourceCollectionResult
from agents.data_collection.smolagents_backend import NotebookGenerationResult


def test_execute_writes_generated_notebook(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "data" / "raw"
    notebook_path = tmp_path / "notebooks" / "eda.ipynb"
    config = {"sources": [{"type": "hf_dataset", "name": "demo"}], "llm": {}}

    agent = DataCollectionAgent(config=config, output_dir=output_dir, notebook_path=notebook_path)

    source_frame = pd.DataFrame({"text": ["hello"], "label": ["pos"]})

    def fake_collect(source):
        return SourceCollectionResult(dataframe=source_frame)

    def fake_generate(*, frame, dataset_path, notebook_path):
        return NotebookGenerationResult(
            notebook={
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# EDA\n"]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": ["import pandas as pd\n", "df = pd.DataFrame()\n"],
                    },
                ],
            },
            success=True,
            notes=["ok"],
        )

    monkeypatch.setattr(agent, "_collect_source", fake_collect)
    monkeypatch.setattr(agent.notebook_backend, "generate_notebook", fake_generate)

    result = agent.execute()

    assert result.dataframe_path == output_dir / "unified_dataset.jsonl"
    assert result.artifacts["eda_notebook"] == str(notebook_path)
    assert result.metadata["eda_notebook_notes"] == ["ok"]
    assert notebook_path.exists()

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
