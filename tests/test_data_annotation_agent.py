import json
from pathlib import Path

import pandas as pd

import agents.data_quality.data_quality_agent as data_quality_agent_module
from agents import AgentResult, DataAnnotationAgent, DataQualityAgent, PipelineRunner
from annotation_agent import AnnotationAgent


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
        self.detect_calls.append(Path(dataset_path))
        return type(
            "Result",
            (),
            {
                "report": {
                    "missing": {"total": 0},
                    "duplicates": 0,
                    "outlier_summary": {"total": 0},
                    "imbalance": {"column": label_column or "label"},
                },
                "success": True,
                "logs": ["detect ok"],
                "attempts": [],
                "notes": [],
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
        self.analyze_calls.append(Path(dataset_path))
        return type(
            "Result",
            (),
            {
                "analysis": {
                    "recommended_strategy": {
                        "missing": "median",
                        "duplicates": "drop",
                        "outliers": "keep",
                    },
                    "task_interpretation": {"task_type": "classification", "primary_modality": "text"},
                    "quality_focus": {
                        "relevant_checks": ["label_distribution"],
                        "irrelevant_checks": [],
                        "priority_actions": ["review_low_confidence_labels"],
                        "notebook_sections": ["label_distribution"],
                        "primary_modality": "text",
                    },
                },
                "success": True,
                "logs": ["analyze ok"],
                "attempts": [],
                "notes": [],
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
        self.fix_calls.append(Path(dataset_path))
        cleaned = pd.read_json(dataset_path, lines=True)
        cleaned.to_json(output_path, orient="records", lines=True)
        return type(
            "Result",
            (),
            {
                "summary": {"input_rows": len(cleaned), "output_rows": len(cleaned)},
                "strategy_used": dict(strategy),
                "justification": "No-op backend for pipeline handoff tests.",
                "success": True,
                "logs": ["fix ok"],
                "attempts": [],
                "notes": [],
            },
        )()

    def compare(self, before_path, after_path, *, analysis_context=None, label_column=None, imbalance_threshold=0.75):
        self.compare_calls.append((Path(before_path), Path(after_path)))
        return type(
            "Result",
            (),
            {
                "comparison": {
                    "missing": {"before_total": 0, "after_total": 0},
                    "duplicates": {"before": 0, "after": 0},
                    "outliers": {"before_total": 0, "after_total": 0},
                    "imbalance": {"after_majority_share": 0.5},
                },
                "success": True,
                "logs": ["compare ok"],
                "attempts": [],
                "notes": [],
            },
        )()


class DummyProcessAdapter:
    def chat(self, messages, json_schema=None):
        payload = json.loads(messages[-1]["content"])
        if "sample_rows" in payload:
            return {"filter_condition": "lang == 'en'", "row_indices": []}
        if "sample_filtered_row" in payload:
            return {
                "row_prompt": (
                    "Translate the value of the 'text' field from English to French. "
                    "Return JSON with the same keys."
                )
            }
        if "rows" in payload:
            translated = {
                "Hello world": "Bonjour le monde",
                "Good morning": "Bonjour",
            }
            rows = []
            for row in payload["rows"]:
                updated = dict(row)
                if updated.get("lang") == "en":
                    updated["text"] = translated.get(updated["text"], updated["text"])
                rows.append(updated)
            return {"rows": rows}
        row = dict(payload)
        translated = {
            "Hello world": "Bonjour le monde",
            "Good morning": "Bonjour",
        }
        if row.get("lang") == "en":
            row["text"] = translated.get(row["text"], row["text"])
        return row


class DummyAnnotationPromptAdapter:
    def chat(self, messages, json_schema=None):
        payload = json.loads(messages[-1]["content"])
        if "sample_rows" in payload:
            return {"filter_condition": None, "row_indices": [0, 1]}
        if "sample_filtered_row" in payload:
            return {
                "row_prompt": (
                    "Solve the problem when possible. If the problem is incomplete and has no solution, "
                    "keep the label empty and set has_complete_problem to false. Otherwise set it to true."
                )
            }
        if "rows" in payload:
            rows = []
            for row in payload["rows"]:
                updated = dict(row)
                text = updated.get("text", "")
                if "Incomplete" in text:
                    updated["label"] = None
                    updated["has_complete_problem"] = False
                else:
                    updated["label"] = "42"
                    updated["has_complete_problem"] = True
                rows.append(updated)
            return {"rows": rows}
        row = dict(payload)
        text = row.get("text", "")
        if "Incomplete" in text:
            row["label"] = None
            row["has_complete_problem"] = False
        else:
            row["label"] = "42"
            row["has_complete_problem"] = True
        return row


class PartialPromptAdapter:
    def chat(self, messages, json_schema=None):
        payload = json.loads(messages[-1]["content"])
        if "sample_rows" in payload:
            return {"filter_condition": None, "row_indices": [0, 1]}
        if "sample_filtered_row" in payload:
            return {
                "row_prompt": (
                    "Solve the problem when possible. If the problem is incomplete and has no solution, "
                    "keep the label empty and set has_complete_problem to false. Otherwise set it to true."
                )
            }
        if "rows" in payload:
            raise RuntimeError("429 Client Error: Too Many Requests")
        row = dict(payload)
        if "dialogue" in row.get("text", "").lower():
            raise RuntimeError("429 Client Error: Too Many Requests")
        row["label"] = "42"
        row["has_complete_problem"] = True
        return row


class DirectPromptOnlyAdapter:
    def chat(self, messages, json_schema=None):
        payload = json.loads(messages[-1]["content"])
        if "sample_rows" in payload or "sample_filtered_row" in payload:
            raise AssertionError("auto_label prompt path should not call selection or prompt-rewrite LLM steps")
        if "rows" in payload:
            rows = []
            for row in payload["rows"]:
                updated = dict(row)
                text = updated.get("text", "")
                if "Incomplete" in text:
                    updated["label"] = None
                    updated["has_complete_problem"] = False
                else:
                    updated["label"] = "42"
                    updated["has_complete_problem"] = True
                rows.append(updated)
            return {"rows": rows}
        row = dict(payload)
        if "Incomplete" in row.get("text", ""):
            row["label"] = None
            row["has_complete_problem"] = False
        else:
            row["label"] = "42"
            row["has_complete_problem"] = True
        return row


def test_annotation_agent_execute_writes_expected_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "data"
    config = {
        "project": {"name": "sentiment-demo", "modality": "text"},
        "agents": {
            "annotation": {
                "task": "sentiment_classification",
                "confidence_threshold": 0.6,
                "classes": [
                    {"name": "positive", "keywords": ["great", "love", "excellent"]},
                    {"name": "negative", "keywords": ["awful", "bad", "hate"]},
                ],
            }
        },
    }
    frame = pd.DataFrame(
        {
            "text": [
                "I love this product",
                "This update is awful",
                "great battery life",
                "ordinary response",
            ],
            "label": [None, "negative", None, None],
            "source": ["reviews"] * 4,
        }
    )
    upstream = AgentResult(dataframe=frame, artifacts={"eda_notebook": str(output_dir / "collection" / "eda.ipynb")})

    agent = DataAnnotationAgent(config=config, output_dir=output_dir)
    result = agent.execute(upstream)

    assert result.dataframe is not None
    assert result.dataframe_path == output_dir / "annotation" / "annotated_dataset.jsonl"
    assert result.dataframe["annotation_auto_label"].tolist()[:3] == ["positive", "negative", "positive"]
    assert "annotation_auto_label_pass_1" in result.dataframe.columns
    assert "annotation_auto_label_pass_2" in result.dataframe.columns
    assert "annotation_intra_agreement" in result.dataframe.columns
    assert result.dataframe["annotation_intra_agreement"].all()
    assert "annotation_confidence" in result.dataframe.columns
    assert Path(result.artifacts["annotation_spec"]).exists()
    assert Path(result.artifacts["annotation_quality"]).exists()
    assert Path(result.artifacts["labelstudio_import"]).exists()
    assert Path(result.artifacts["low_confidence_review"]).exists()
    assert result.metadata["annotation"]["task"] == "sentiment_classification"
    assert result.metrics["annotation_low_confidence_count"] >= 1

    quality_report = json.loads(Path(result.artifacts["annotation_quality"]).read_text(encoding="utf-8"))
    assert quality_report["label_dist"]["negative"] >= 1
    assert quality_report["intra_agreement_rate"] == 1.0

    labelstudio_payload = json.loads(Path(result.artifacts["labelstudio_import"]).read_text(encoding="utf-8"))
    assert labelstudio_payload[0]["annotations"][0]["result"][0]["value"]["choices"] == ["positive"]

    review_payload = json.loads(Path(result.artifacts["low_confidence_review"]).read_text(encoding="utf-8"))
    assert any(task["meta"]["review_reason"] == "low_confidence" for task in review_payload)

    spec_text = Path(result.artifacts["annotation_spec"]).read_text(encoding="utf-8")
    assert "sentiment_classification" in spec_text
    assert "positive" in spec_text
    assert spec_text.count("I love this product") >= 2


def test_annotation_agent_integrates_with_quality_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_quality_agent_module, "SmolagentsQualityBackend", DummyQualityBackend)
    output_dir = tmp_path / "data"
    notebook_path = output_dir / "collection" / "eda.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}),
        encoding="utf-8",
    )
    config = {
        "project": {"modality": "text"},
        "agents": {
            "annotation": {
                "classes": [
                    {"name": "positive", "keywords": ["great", "love"]},
                    {"name": "negative", "keywords": ["bad", "awful"]},
                ]
            }
        },
    }
    frame = pd.DataFrame({"text": ["great result", "bad output"], "label": [None, None], "source": ["demo", "demo"]})

    runner = PipelineRunner(
        [
            DataAnnotationAgent(config=config, output_dir=output_dir),
            DataQualityAgent(config=config, output_dir=output_dir, notebook_path=notebook_path),
        ]
    )
    result = runner.run({"dataframe": frame})

    assert result.dataframe_path == output_dir / "quality" / "cleaned_dataset.jsonl"
    assert result.artifacts["annotated_dataset"] == str(output_dir / "annotation" / "annotated_dataset.jsonl")
    assert Path(result.artifacts["quality_report"]).exists()
    assert result.metadata["annotation"]["quality"]["label_dist"]["positive"] == 1
    assert result.metadata["quality_report"]["missing"]["total"] == 0

    backend = runner.agents[1].backend
    assert backend.detect_calls[0] == output_dir / "annotation" / "annotated_dataset.jsonl"


def test_annotation_agent_shim_matches_repo_agent() -> None:
    assert AnnotationAgent is DataAnnotationAgent


def test_process_applies_llm_selected_row_transformations(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "translations.csv"
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["Hello world", "Salut", "Good morning"],
            "lang": ["en", "fr", "en"],
        }
    )
    frame.to_csv(input_path, index=False)
    agent = DataAnnotationAgent(
        config={
            "agents": {
                "annotation": {
                    "process_config": {
                        "parallel_workers": 4,
                        "max_retries": 2,
                    }
                }
            }
        },
        output_dir=tmp_path,
    )
    monkeypatch.setattr(agent, "_build_model_adapter", lambda: DummyProcessAdapter())

    output_path = agent.process(input_path, "Translate all English texts to French")

    assert output_path == tmp_path / "translations_processed.csv"
    processed = pd.read_csv(output_path)
    assert processed["text"].tolist() == ["Bonjour le monde", "Salut", "Bonjour"]
    assert processed["lang"].tolist() == ["en", "fr", "en"]


def test_annotation_prompt_from_config_can_add_new_fields(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "Complete problem with enough detail",
                "Incomplete fragment without enough detail",
                "Already solved problem",
            ],
            "label": [None, None, "7"],
            "source": ["math", "math", "math"],
            "collected_at": pd.to_datetime(
                ["2026-03-19T00:00:00Z", "2026-03-19T00:05:00Z", "2026-03-19T00:10:00Z"],
                utc=True,
            ),
        }
    )
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "prompt": (
                        "For rows with no label, try to solve the text problem. "
                        "If a problem is incomplete (and has no solution), keep label empty, "
                        'set new field "has_complete_problem" to false, otherwise to true.'
                    )
                }
            },
        },
        output_dir=output_dir,
    )
    monkeypatch.setattr(agent, "_build_model_adapter", lambda: DummyAnnotationPromptAdapter())

    result = agent.execute({"dataframe": frame})

    assert result.dataframe is not None
    assert result.metadata["annotation"]["prompt"] is not None
    assert "has_complete_problem" in result.dataframe.columns
    assert result.dataframe.loc[0, "label"] == "42"
    assert bool(result.dataframe.loc[0, "has_complete_problem"]) is True
    assert pd.isna(result.dataframe.loc[1, "label"])
    assert bool(result.dataframe.loc[1, "has_complete_problem"]) is False
    assert bool(result.dataframe.loc[1, "annotation_needs_review"]) is False
    assert result.dataframe.loc[2, "label"] == "7"
    assert result.dataframe.loc[2, "annotation_label_origin"] == "existing"


def test_annotation_quality_reports_inter_expert_and_intra_agreement(tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "great result and clear answer",
                "bad output and broken flow",
                "great product and fast response",
                "awful delay and bad support",
            ],
            "label": ["positive", "negative", "positive", "negative"],
            "source": ["demo"] * 4,
        }
    )
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "classes": [
                        {"name": "positive", "keywords": ["great", "fast"]},
                        {"name": "negative", "keywords": ["bad", "awful", "broken"]},
                    ]
                }
            },
        },
        output_dir=output_dir,
    )

    result = agent.execute({"dataframe": frame})

    assert result.metadata["annotation"]["quality"]["intra_agreement_rate"] == 1.0
    assert result.metadata["annotation"]["quality"]["inter_expert_agreement_rate"] == 1.0
    assert result.metadata["annotation"]["quality"]["inter_expert_kappa"] == 1.0


def test_annotation_prompt_adds_declared_fields_even_for_prelabeled_rows(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "Complete problem with enough detail",
                "Incomplete fragment without enough detail",
            ],
            "label": ["42", "17"],
            "source": ["math", "math"],
        }
    )
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "prompt": (
                        "For rows with no label, try to solve the text problem. "
                        'Set new field "has_complete_problem" to false for incomplete rows and true otherwise.'
                    )
                }
            },
        },
        output_dir=output_dir,
    )
    monkeypatch.setattr(agent, "_build_model_adapter", lambda: DummyAnnotationPromptAdapter())

    result = agent.execute({"dataframe": frame})

    assert "has_complete_problem" in result.dataframe.columns
    assert bool(result.dataframe.loc[0, "has_complete_problem"]) is True
    assert bool(result.dataframe.loc[1, "has_complete_problem"]) is False
    assert result.dataframe.loc[0, "label"] == "42"
    assert result.dataframe.loc[1, "label"] == "17"


def test_annotation_agent_prefers_cleaned_dataset_when_run_after_quality(tmp_path) -> None:
    output_dir = tmp_path / "data"
    (output_dir / "collection").mkdir(parents=True, exist_ok=True)
    (output_dir / "quality").mkdir(parents=True, exist_ok=True)
    unified = pd.DataFrame({"text": ["raw"], "label": ["before"], "source": ["demo"]})
    cleaned = pd.DataFrame({"text": ["clean"], "label": ["after"], "source": ["demo"]})
    unified.to_json(output_dir / "collection" / "unified_dataset.jsonl", orient="records", lines=True)
    cleaned.to_json(output_dir / "quality" / "cleaned_dataset.jsonl", orient="records", lines=True)

    agent = DataAnnotationAgent(config={"project": {"modality": "text"}}, output_dir=output_dir)
    resolved = agent._resolve_dataframe(None)

    assert resolved.equals(cleaned)


def test_prompt_rows_that_fail_model_processing_stay_unresolved(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "text": [
                "In this dialogue choose the right statement. A) One B) Two C) Three",
                "Find the value of x if x + 2 = 5",
            ],
            "label": [None, None],
            "source": ["demo", "demo"],
        }
    )
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "prompt": (
                        "For rows with no label, try to solve the text problem. "
                        "If a problem is incomplete (and has no solution), keep label empty, "
                        'set new field "has_complete_problem" to false, otherwise to true.'
                    )
                }
            },
        },
        output_dir=tmp_path,
    )
    monkeypatch.setattr(agent, "_build_model_adapter", lambda: PartialPromptAdapter())

    result = agent.execute({"dataframe": frame})

    assert result.dataframe is not None
    assert pd.isna(result.dataframe.loc[0, "label"])
    assert result.dataframe.loc[0, "annotation_label_origin"] == "unresolved"
    assert bool(result.dataframe.loc[0, "annotation_needs_review"]) is True
    assert bool(result.dataframe.loc[1, "has_complete_problem"]) is True
    assert result.dataframe.loc[1, "label"] == "42"
    assert result.dataframe.loc[1, "annotation_label_origin"] == "prompt"


def test_annotation_agent_caps_parallelism_for_remote_models(tmp_path) -> None:
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "llm": {
                "model": "remote-model",
                "base_url": "https://openrouter.ai/api/v1/chat/completions",
            },
            "agents": {
                "annotation": {
                    "process_config": {
                        "parallel_workers": 8,
                    }
                }
            },
        },
        output_dir=tmp_path,
    )

    assert agent._effective_parallel_workers() == 2


def test_annotation_agent_root_level_base_url_overrides_global_llm(tmp_path) -> None:
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "llm": {
                "model": "global-model",
                "base_url": "https://shared.example/v1/chat/completions",
            },
            "agents": {
                "annotation": {
                    "base_url": "https://annotation.example/v1/chat/completions",
                }
            },
        },
        output_dir=tmp_path,
    )

    assert agent.process_config["base_url"] == "https://annotation.example/v1/chat/completions"


def test_annotation_agent_uses_safe_default_max_tokens(tmp_path) -> None:
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "llm": {
                "model": "local-model",
                "base_url": "http://localhost:8000/v1/chat/completions",
            },
        },
        output_dir=tmp_path,
    )

    assert agent.process_config["max_tokens"] == 8192


def test_annotation_prompt_uses_direct_row_transform_path(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "Complete problem with enough detail",
                "Incomplete fragment without enough detail",
            ],
            "label": [None, None],
            "source": ["math", "math"],
        }
    )
    agent = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "prompt": (
                        "For rows with no label, try to solve the text problem. "
                        "If a problem is incomplete (and has no solution), keep label empty, "
                        'set new field "has_complete_problem" to false, otherwise to true.'
                    )
                }
            },
        },
        output_dir=output_dir,
    )
    monkeypatch.setattr(agent, "_build_model_adapter", lambda: DirectPromptOnlyAdapter())

    result = agent.execute({"dataframe": frame})

    assert result.dataframe is not None
    assert result.dataframe.loc[0, "label"] == "42"
    assert bool(result.dataframe.loc[0, "has_complete_problem"]) is True
    assert pd.isna(result.dataframe.loc[1, "label"])
    assert bool(result.dataframe.loc[1, "has_complete_problem"]) is False
    assert bool(result.dataframe.loc[1, "annotation_needs_review"]) is False


def test_annotation_agent_does_not_guess_high_cardinality_numeric_labels(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "text": [f"problem {index}" for index in range(1, 36)],
            "label": [str(index) for index in range(1, 31)] + [None] * 5,
            "source": ["demo"] * 35,
        }
    )
    agent = DataAnnotationAgent(config={"project": {"modality": "text"}}, output_dir=tmp_path)

    labeled = agent.auto_label(frame)

    assert labeled["label"].iloc[:30].tolist() == [str(index) for index in range(1, 31)]
    assert labeled["label"].iloc[30:].isna().all()
    assert set(labeled["annotation_label_origin"].iloc[30:].tolist()) == {"unresolved"}
