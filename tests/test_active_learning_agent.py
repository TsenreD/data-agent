from pathlib import Path
import pickle

import pandas as pd

from agents import ActiveLearningAgent, DataAnnotationAgent, PipelineRunner


class DummyAnnotationPromptAdapter:
    def chat(self, messages, json_schema=None):
        payload = __import__("json").loads(messages[-1]["content"])
        if "sample_rows" in payload:
            return {"filter_condition": None, "row_indices": [0, 1]}
        if "sample_filtered_row" in payload:
            return {
                "row_prompt": (
                    "If the problem statement is complete, set has_complete_problem to true and fill label. "
                    "If it is incomplete, set has_complete_problem to false and leave label empty."
                )
            }
        rows = payload.get("rows")
        if rows is None:
            rows = [payload]
        result_rows = []
        for row in rows:
            updated = dict(row)
            text = str(updated.get("text", ""))
            if "Incomplete" in text:
                updated["label"] = None
                updated["has_complete_problem"] = False
            else:
                updated["label"] = "42"
                updated["has_complete_problem"] = True
            result_rows.append(updated)
        if "rows" in payload:
            return {"rows": result_rows}
        return result_rows[0]


def test_active_learning_agent_selects_prompt_aligned_target_and_queries_pool(tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "Complete problem about algebra with all conditions",
                "Incomplete fragment missing the question",
                "Complete geometry problem with a diagram description",
                "Incomplete problem statement without numbers",
                "Complete probability problem with enough context",
                "Incomplete prompt with only answer choices",
                "Unlabeled draft that looks complete but needs review",
                "Unlabeled short fragment with missing context",
            ],
            "notes": [
                "math",
                "math",
                "math",
                "math",
                "math",
                "math",
                "draft",
                "draft",
            ],
            "has_complete_problem": [True, False, True, False, True, False, None, None],
            "label": ["1", None, "2", None, "3", None, None, None],
        }
    )
    agent = ActiveLearningAgent(
        config={
            "agents": {
                "active_learning": {
                    "task_prompt": "Train classification model to determine whether text contains a full problem, or an incomplete one",
                    "batch_size": 2,
                    "model": {"epochs": 18, "dim": 32, "bucket_size": 2048},
                    "docker": {"enabled": False},
                }
            }
        },
        output_dir=output_dir,
    )

    result = agent.execute({"dataframe": frame})

    assert result.dataframe is not None
    assert result.metadata["active_learning"]["target_column"] == "has_complete_problem"
    assert "text" in result.metadata["active_learning"]["feature_columns"]
    assert result.metadata["active_learning"]["selected_rows"] == 2
    assert result.dataframe["active_learning_selected"].sum() == 2
    assert Path(result.artifacts["active_learning_queries"]).exists()
    assert Path(result.artifacts["active_learning_summary"]).exists()
    assert Path(result.artifacts["active_learning_report"]).exists()
    assert Path(result.artifacts["active_learning_train_script"]).exists()
    assert Path(result.artifacts["active_learning_train_dataset"]).exists()
    assert Path(result.artifacts["active_learning_val_dataset"]).exists()
    assert Path(result.artifacts["active_learning_pool_dataset"]).exists()
    assert Path(result.artifacts["active_learning_model"]).exists()
    assert Path(result.artifacts["active_learning_training_metrics"]).exists()
    assert result.metrics["active_learning_accuracy"] >= 0.0


def test_active_learning_run_cycle_and_report(tmp_path) -> None:
    agent = ActiveLearningAgent(
        config={
            "agents": {
                "active_learning": {
                    "task_prompt": "Classify whether a text is complete or incomplete",
                    "batch_size": 2,
                    "model": {"epochs": 20, "dim": 24, "bucket_size": 1024},
                    "docker": {"enabled": False},
                }
            }
        },
        output_dir=tmp_path / "data",
    )
    labeled = pd.DataFrame(
        {
            "text": [
                "Complete problem with all data",
                "Incomplete fragment",
                "Complete statement with conditions",
                "Incomplete prompt with missing variables",
                "Complete proof task",
                "Incomplete sentence without the actual question",
            ],
            "has_complete_problem": [True, False, True, False, True, False],
        }
    )
    pool = pd.DataFrame(
        {
            "text": [
                "Complete olympiad problem statement",
                "Incomplete stub with no task",
                "Complete combinatorics setup",
                "Incomplete line copied from a worksheet",
            ],
            "has_complete_problem": [True, False, True, False],
        }
    )

    history = agent.run_cycle(labeled, pool, strategy="entropy", n_iterations=2, batch_size=2)
    random_history = agent.run_cycle(labeled, pool, strategy="random", n_iterations=2, batch_size=2)
    report_path = agent.report(history, random_history=random_history)

    assert len(history) == 2
    assert history[0]["n_labeled"] == 6
    assert history[1]["n_labeled"] == 8
    assert report_path.exists()


def test_active_learning_integrates_after_annotation_prompt(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "text": [
                "Complete problem with enough detail",
                "Incomplete fragment without enough detail",
                "Complete number theory problem",
                "Incomplete text that references a missing diagram",
                "Already solved complete problem",
                "Another incomplete problem fragment",
            ],
            "label": [None, None, None, None, "7", None],
            "source": ["math"] * 6,
        }
    )
    annotation = DataAnnotationAgent(
        config={
            "project": {"modality": "text"},
            "agents": {
                "annotation": {
                    "prompt": (
                        "For rows with no label, solve complete problems when possible. "
                        'Set "has_complete_problem" to false for incomplete rows and true otherwise.'
                    )
                },
                "active_learning": {
                    "task_prompt": "Train classification model to determine whether text contains a full problem, or an incomplete one",
                    "batch_size": 2,
                    "feature_columns": ["text"],
                    "target_column": "label",
                    "model": {"epochs": 16, "dim": 24, "bucket_size": 1024},
                    "docker": {"enabled": False},
                },
            },
        },
        output_dir=output_dir,
    )
    monkeypatch.setattr(annotation, "_build_model_adapter", lambda: DummyAnnotationPromptAdapter())
    active = ActiveLearningAgent(config=annotation.config, output_dir=output_dir)

    runner = PipelineRunner([annotation, active])
    result = runner.run({"dataframe": frame})

    assert result.metadata["active_learning"]["target_column"] == "label"
    assert result.artifacts["annotated_dataset"] == str(output_dir / "annotation" / "annotated_dataset.jsonl")
    assert Path(result.artifacts["active_learning_dataset"]).exists()
    assert Path(result.artifacts["active_learning_model"]).exists()


def test_active_learning_prefers_docker_training_when_enabled(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "text": [
                "complete problem statement",
                "incomplete fragment",
                "complete theorem statement",
                "incomplete draft line",
            ],
            "is_complete": [True, False, True, False],
        }
    )
    agent = ActiveLearningAgent(
        config={
            "agents": {
                "active_learning": {
                    "task_prompt": "Classify complete vs incomplete text",
                    "feature_columns": ["text"],
                    "target_column": "is_complete",
                    "batch_size": 2,
                    "docker": {"enabled": True},
                }
            }
        },
        output_dir=tmp_path / "data",
    )

    calls = {"docker": 0, "local": 0}

    def fake_docker_training(artifacts):
        calls["docker"] += 1
        checkpoint = {
            "architecture": "numpy_softmax",
            "vectorizer": {"token_to_index": {"complete": 1, "incomplete": 2}, "unknown_index": 0},
            "index_to_label": ["false", "true"],
            "weights": [[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]],
            "bias": [0.0, 0.0],
        }
        with Path(artifacts.model_path).open("wb") as handle:
            pickle.dump(checkpoint, handle)
        Path(artifacts.training_metrics_path).write_text(
            '{"accuracy": 1.0, "macro_f1": 1.0, "evaluated_rows": 2, "train_rows": 2, "val_rows": 2, "backend": "torch"}',
            encoding="utf-8",
        )

    def fail_local_training(_):
        calls["local"] += 1
        raise AssertionError("local training should not be used when docker is enabled")

    monkeypatch.setattr(agent, "_run_training_in_docker", fake_docker_training)
    monkeypatch.setattr(agent, "_run_training_locally", fail_local_training)

    result = agent.execute({"dataframe": frame})
    assert calls["docker"] == 1
    assert calls["local"] == 0
    assert result.metadata["active_learning"]["metrics"]["backend"] == "torch"
