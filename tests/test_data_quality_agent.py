import json
import sys
from pathlib import Path

import pandas as pd

import agents.data_quality.data_quality_agent as data_quality_agent_module
from agents import AgentResult, DataQualityAgent


class DummyQualityBackend:
    def __init__(self, *args, **kwargs) -> None:
        self.detect_calls = []
        self.analyze_calls = []
        self.fix_calls = []
        self.compare_calls = []

    def detect_issues(
        self,
        dataset_path,
        *,
        project_context=None,
        label_column=None,
        imbalance_threshold=0.75,
        preview_frame=None,
    ):
        self.detect_calls.append(
            (Path(dataset_path), dict(project_context or {}), label_column, imbalance_threshold, preview_frame)
        )
        return type(
            "Result",
            (),
            {
                "report": {
                    "missing": {"total": 2},
                    "duplicates": 1,
                    "outlier_summary": {"total": 1},
                    "imbalance": {"column": "label"},
                    "text_profile": {"empty_rows": 0, "duplicate_text_rows": 0},
                },
                "success": True,
                "logs": ["detect ok"],
                "attempts": [],
                "notes": ["detected by agent"],
            },
        )()

    def analyze(
        self,
        dataset_path,
        *,
        detection_report,
        project_context=None,
        task_description="",
        strategy_hint=None,
        label_column=None,
        preview_frame=None,
    ):
        self.analyze_calls.append(
            (
                Path(dataset_path),
                detection_report,
                dict(project_context or {}),
                task_description,
                dict(strategy_hint or {}),
                label_column,
                None if preview_frame is None else list(preview_frame.columns),
            )
        )
        return type(
            "Result",
            (),
            {
                "analysis": {
                    "task_interpretation": {
                        "task_type": "regression_like_math_answers",
                        "label_semantics": "numeric answer target for math tasks",
                    },
                    "recommended_strategy": {
                        "missing": "median",
                        "duplicates": "drop",
                        "outliers": "keep",
                    },
                    "alternative_strategies": [
                        {"missing": "median", "duplicates": "drop", "outliers": "keep"},
                        {"missing": "drop", "duplicates": "drop", "outliers": "keep"},
                    ],
                    "quality_focus": {
                        "relevant_checks": ["missing_text", "duplicate_problems", "normalized_text_duplicates"],
                        "irrelevant_checks": ["label_distribution"],
                        "priority_columns": ["text", "metadata"],
                        "notebook_sections": ["missingness", "text_length_distribution", "source_label_coverage"],
                        "priority_actions": ["drop_all_null_optional_columns", "deduplicate_normalized_text", "drop_rows_missing_label"],
                        "primary_modality": "text",
                        "dedup_rationale": "Repeated prompts should be removed before considering row drops.",
                    },
                    "justification": "Numeric answer labels are not useful for class-balance analysis here.",
                },
                "success": True,
                "logs": ["analyze ok"],
                "attempts": [],
                "notes": ["analyzed by agent"],
            },
        )()

    def fix(
        self,
        dataset_path,
        output_path,
        *,
        strategy,
        task_description="",
        analysis_context=None,
        label_column=None,
        imbalance_threshold=0.75,
    ):
        self.fix_calls.append(
            (Path(dataset_path), Path(output_path), dict(strategy), task_description, analysis_context, label_column)
        )
        cleaned = pd.DataFrame({"text": ["alpha", "beta"], "label": ["yes", "no"], "score": [10, 20]})
        cleaned.to_json(output_path, orient="records", lines=True)
        return type(
            "Result",
            (),
            {
                "summary": {"input_rows": 3, "output_rows": 2},
                "strategy_used": dict(strategy),
                "justification": "Agent selected the configured cleanup strategy after inspecting the dataset.",
                "success": True,
                "logs": ["fix ok"],
                "attempts": [],
                "notes": ["cleaned by agent"],
            },
        )()

    def compare(self, before_path, after_path, *, analysis_context=None, label_column=None, imbalance_threshold=0.75):
        self.compare_calls.append((Path(before_path), Path(after_path), analysis_context, label_column, imbalance_threshold))
        return type(
            "Result",
            (),
            {
                "comparison": {
                    "missing": {"before_total": 2, "after_total": 0},
                    "duplicates": {"before": 1, "after": 0},
                    "outliers": {"before_total": 1, "after_total": 0},
                    "imbalance": {"after_majority_share": 0.5},
                },
                "success": True,
                "logs": ["compare ok"],
                "attempts": [],
                "notes": ["compared by agent"],
            },
        )()


def test_detect_issues_delegates_to_agentic_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    agent = DataQualityAgent(config={"project": {"modality": "text"}}, output_dir=tmp_path)
    frame = pd.DataFrame({"text": ["alpha"], "label": ["yes"]})

    report = agent.detect_issues(frame)

    assert report["missing"]["total"] == 2
    assert agent.backend.detect_calls
    assert agent.backend.detect_calls[0][1]["modality"] == "text"


def test_fix_delegates_to_agentic_backend_and_reads_cleaned_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    agent = DataQualityAgent(config={}, output_dir=tmp_path)
    frame = pd.DataFrame({"text": ["alpha"], "label": ["yes"], "score": [10]})

    cleaned = agent.fix(frame, strategy={"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"})

    assert len(cleaned) == 2
    assert list(cleaned.columns) == ["text", "label", "score"]
    assert agent.backend.fix_calls[0][2]["duplicates"] == "drop"


def test_compare_delegates_to_agentic_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    agent = DataQualityAgent(config={}, output_dir=tmp_path)
    before = pd.DataFrame({"text": ["alpha"], "label": ["yes"]})
    after = pd.DataFrame({"text": ["beta"], "label": ["no"]})

    comparison = agent.compare(before, after)

    assert comparison["duplicates"]["after"] == 0
    assert agent.backend.compare_calls


def test_execute_consumes_upstream_agent_result_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    output_dir = tmp_path / "data"
    notebook_path = output_dir / "collection" / "eda.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# EDA\n"]}],
            }
        ),
        encoding="utf-8",
    )
    raw_frame = pd.DataFrame({"text": ["alpha"], "label": ["yes"], "score": [10]})
    raw_path = output_dir / "collection" / "unified_dataset.jsonl"
    raw_frame.to_json(raw_path, orient="records", lines=True)
    upstream = AgentResult(
        dataframe=raw_frame,
        dataframe_path=raw_path,
        artifacts={"eda_notebook": str(notebook_path)},
        metadata={"failed_sources": []},
        metrics={"row_count": len(raw_frame)},
        logs=["collection log"],
    )

    agent = DataQualityAgent(config={}, output_dir=output_dir, notebook_path=notebook_path)
    result = agent.execute(upstream)

    assert result.dataframe is not None
    assert result.dataframe_path == output_dir / "quality" / "cleaned_dataset.jsonl"
    assert Path(result.artifacts["quality_report"]).exists()
    assert Path(result.artifacts["quality_analysis"]).exists()
    assert Path(result.artifacts["quality_comparison"]).exists()
    assert Path(result.artifacts["cleaned_dataset"]).exists()
    assert Path(result.artifacts["eda_notebook"]).exists()
    assert result.metadata["quality_strategy"]["outliers"] == "keep"
    assert result.metadata["quality_decision"]["mode"] == "automatic"
    assert result.metadata["quality_justification"]
    assert result.metadata["quality_analysis"]["quality_focus"]["irrelevant_checks"] == ["label_distribution"]
    assert result.metadata["quality_analysis"]["quality_focus"]["priority_actions"] == [
        "drop_all_null_optional_columns",
        "deduplicate_normalized_text",
        "drop_rows_missing_label",
    ]
    assert agent.backend.analyze_calls

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert any("Data Quality Review" in "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_execute_without_payload_uses_saved_unified_dataset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    output_dir = tmp_path / "data"
    notebook_path = output_dir / "collection" / "eda.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}),
        encoding="utf-8",
    )
    raw_frame = pd.DataFrame({"text": ["alpha"], "label": ["yes"], "score": [10]})
    raw_path = output_dir / "collection" / "unified_dataset.jsonl"
    raw_frame.to_json(raw_path, orient="records", lines=True)

    agent = DataQualityAgent(config={}, output_dir=output_dir, notebook_path=notebook_path)
    result = agent.execute()

    assert result.dataframe is not None
    assert agent.backend.detect_calls[0][0] == raw_path


def test_execute_prompts_user_to_choose_strategy(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    output_dir = tmp_path / "data"
    notebook_path = output_dir / "collection" / "eda.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}),
        encoding="utf-8",
    )
    raw_frame = pd.DataFrame({"text": ["alpha"], "label": ["yes"], "score": [10]})
    raw_path = output_dir / "collection" / "unified_dataset.jsonl"
    raw_frame.to_json(raw_path, orient="records", lines=True)

    agent = DataQualityAgent(config={}, output_dir=output_dir, notebook_path=notebook_path)
    result = agent.execute()

    captured = capsys.readouterr()
    assert "Data Quality Analyzer Suggestions" in captured.out
    assert result.metadata["quality_decision"]["mode"] == "human_in_the_loop"
    assert result.metadata["quality_decision"]["selected_option"] == 2
    assert agent.backend.fix_calls[0][2]["missing"] == "drop"
