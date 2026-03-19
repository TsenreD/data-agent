from pathlib import Path

import pandas as pd

from agents.data_quality.smolagents_backend import SmolagentsQualityBackend


def test_quality_backend_builds_detect_task_with_dataset_paths() -> None:
    backend = SmolagentsQualityBackend(model=object())
    task = backend._build_detect_task(
        Path("/tmp/input.jsonl"),
        project_context={"modality": "text"},
        label_column="label",
        imbalance_threshold=0.75,
        preview_frame=pd.DataFrame({"text": ["sample"], "label": [1]}),
    )

    assert "Sandbox dataset path:" in task
    assert "Primary modality: text" in task
    assert "Preferred label column: label" in task
    assert "Imbalance threshold: 0.75" in task
    assert "text length distribution" in task
    assert "row sparsity" in task
    assert "deduplication as a core quality decision" in task


def test_quality_backend_builds_fix_task_with_strategy_and_output_path(tmp_path) -> None:
    backend = SmolagentsQualityBackend(model=object())
    task = backend._build_fix_task(
        tmp_path / "input.jsonl",
        tmp_path / "output.jsonl",
        strategy={"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"},
        task_description="classification",
        analysis_context={"quality_focus": {"irrelevant_checks": ["label_distribution"]}},
        label_column="label",
        imbalance_threshold=0.8,
    )

    assert "Sandbox output dataset path:" in task
    assert '"missing": "median"' in task
    assert "Task description: classification" in task
    assert "Task-aware analysis context:" in task
    assert "Do not merely describe the cleaning steps; execute them" in task
    assert "Do not drop rows only because `label` is null" in task
    assert "maximize retention of useful data" in task
    assert "Perform deduplication early" in task


def test_quality_backend_builds_analyze_task_with_report_and_project_context(tmp_path) -> None:
    backend = SmolagentsQualityBackend(model=object())
    task = backend._build_analyze_task(
        tmp_path / "input.jsonl",
        detection_report={"missing": {"total": 2}},
        project_context={"name": "math-problems", "modality": "text"},
        task_description="predict numeric answers from olympiad problems",
        strategy_hint={"missing": "median"},
        label_column="label",
        preview_frame=pd.DataFrame({"text": ["a"], "label": [1]}),
    )

    assert "Detected issue report:" in task
    assert "Project context:" in task
    assert "Primary modality: text" in task
    assert "alternative_strategies" in task
    assert "Preview rows:" in task
    assert "predict numeric answers from olympiad problems" in task
    assert "text_actions" in task
    assert "Missing labels are not an automatic reason to drop rows" in task
    assert "optimize for maximum useful data retention" in task
    assert "Make deduplication policy explicit" in task


def test_quality_backend_passes_empty_tool_list_to_code_agent(monkeypatch, tmp_path) -> None:
    backend = SmolagentsQualityBackend(model=object())
    captured = {}

    def fake_run_agent(**kwargs):
        captured["tools"] = kwargs["tools"]
        captured["imports"] = kwargs["additional_imports"]
        captured["max_steps"] = kwargs["max_steps"]
        return type(
            "RunResult",
            (),
            {
                "output": '{"report":{"row_count":1,"missing":{"total":0},"duplicates":0,"outlier_summary":{"total":0},"imbalance":{"column":"label"}},"notes":["ok"]}',
                "state": "done",
                "steps": [],
            },
        )()

    monkeypatch.setattr(backend, "_run_agent", fake_run_agent)

    result = backend.detect_issues(
        tmp_path / "input.jsonl",
        project_context={"modality": "text"},
        label_column="label",
        preview_frame=pd.DataFrame({"text": ["sample"], "label": [1]}),
    )

    assert result.success is True
    assert result.report is not None
    assert captured["tools"] == []
    assert "pandas" in captured["imports"]


def test_quality_backend_parse_fix_output_accepts_plain_json() -> None:
    payload, notes = SmolagentsQualityBackend._parse_fix_output(
        '{"summary":{"input_rows":3,"output_rows":2},"strategy_used":{"duplicates":"drop"},"justification":"ok","notes":["done"]}'
    )

    assert payload["summary"]["output_rows"] == 2
    assert payload["strategy_used"]["duplicates"] == "drop"
    assert payload["justification"] == "ok"
    assert notes == ["done"]


def test_quality_backend_parse_analyze_output_accepts_plain_json() -> None:
    payload, notes = SmolagentsQualityBackend._parse_analyze_output(
        '{"analysis":{"task_interpretation":{"task_type":"regression"},"recommended_strategy":{"duplicates":"drop"},"alternative_strategies":[{"duplicates":"drop"},{"duplicates":"keep"}],"quality_focus":{"irrelevant_checks":["label_distribution"]},"justification":"ok"},"notes":["done"]}'
    )

    assert payload["analysis"]["task_interpretation"]["task_type"] == "regression"
    assert payload["analysis"]["recommended_strategy"]["duplicates"] == "drop"
    assert notes == ["done"]


def test_quality_backend_parse_analyze_output_normalizes_string_task_interpretation() -> None:
    payload, _ = SmolagentsQualityBackend._parse_analyze_output(
        '{"analysis":{"task_interpretation":"math qa regression task","recommended_strategy":{"duplicates":"drop"},"quality_focus":{"primary_modality":"text"},"justification":"ok"}}'
    )

    assert payload["analysis"]["task_interpretation"]["summary"] == "math qa regression task"
    assert payload["analysis"]["quality_focus"]["primary_modality"] == "text"
