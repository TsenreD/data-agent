import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

from ..base import AgentResult, BaseAgent

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TEXT_COLUMN_HINTS = {
    "text",
    "problem",
    "question",
    "prompt",
    "statement",
    "body",
    "content",
    "description",
    "title",
}
DEFAULT_QUERY_STRATEGY = "entropy"
DEFAULT_BATCH_SIZE = 20
DEFAULT_REPORT_PATH = "active_learning_curve.png"
DEFAULT_SCRIPT_PATH = "train_active_learning.py"
DEFAULT_MODEL_PATH = "model.pth"
DEFAULT_TRAINING_METRICS_PATH = "training_metrics.json"
DEFAULT_AGENT_MAX_STEPS = 8
DEFAULT_ANNOTATION_TARGET_COLUMN = "annotation_label"


@dataclass(slots=True)
class ActiveLearningArtifacts:
    dataset_path: Path
    queries_path: Path
    summary_path: Path
    report_path: Path
    train_dataset_path: Path
    val_dataset_path: Path
    pool_dataset_path: Path
    train_script_path: Path
    model_path: Path
    training_metrics_path: Path


@dataclass(slots=True)
class TaskSelection:
    feature_columns: list[str]
    target_column: str
    task_prompt: str


def _compose_texts_from_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    task_prompt: str | None = None,
) -> list[str]:
    prompt = (task_prompt or "").strip()
    prefix = f"Task: {prompt}\n" if prompt else ""
    texts: list[str] = []
    for _, row in frame.iterrows():
        chunks: list[str] = []
        for column in feature_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            chunks.append(str(value))
        body = "\n".join(chunks)
        texts.append(prefix + body if prefix else body)
    return texts


def _normalize_label_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value).strip()


class ActiveLearningAgent(BaseAgent):
    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        output_dir: str | Path = "data",
        task_prompt: str | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.base_output_dir = Path(output_dir)
        self.output_dir = self._stage_output_dir(self.base_output_dir, "active_learning")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_config = self._resolve_active_learning_config()
        self.task_prompt = self._normalize_text(task_prompt) or self._normalize_text(
            self.active_config.get("task_prompt") or self.active_config.get("prompt")
        )
        self.query_strategy = str(self.active_config.get("strategy", DEFAULT_QUERY_STRATEGY)).strip() or DEFAULT_QUERY_STRATEGY
        self.batch_size = int(self.active_config.get("batch_size", DEFAULT_BATCH_SIZE))
        self.test_size = float(self.active_config.get("test_size", 0.25))
        self.random_seed = int(self.active_config.get("random_seed", 13))
        self.agent_max_steps = int(self.active_config.get("max_steps", DEFAULT_AGENT_MAX_STEPS))

    def run(self, dataframe: pd.DataFrame, task_prompt: str | None = None) -> pd.DataFrame:
        result = self.execute({"dataframe": dataframe, "task_prompt": task_prompt or self.task_prompt})
        if result.dataframe is None:
            raise RuntimeError("ActiveLearningAgent did not produce a dataframe.")
        return result.dataframe

    def execute(self, payload: Any | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting ActiveLearningAgent execution.", logs)

        upstream = payload if isinstance(payload, AgentResult) else None
        frame = self._resolve_dataframe(payload)
        selection = self._select_task_columns(frame, payload)

        target_series = frame[selection.target_column]
        labeled_mask = target_series.notna()
        pool_mask = ~labeled_mask
        labeled = frame.loc[labeled_mask].copy()
        pool = frame.loc[pool_mask].copy()
        if labeled.empty:
            raise ValueError(f"ActiveLearningAgent requires labeled rows in `{selection.target_column}`.")
        class_count = labeled[selection.target_column].dropna().apply(_normalize_label_value).nunique()
        if class_count < 2:
            raise ValueError(
                f"ActiveLearningAgent requires at least two classes in `{selection.target_column}` after null filtering."
            )

        train_df, val_df = self._split_train_test(labeled, selection.target_column)
        prepared = self._prepare_training_files(train_df, val_df, pool, selection)
        metrics = self._run_generated_training(prepared, logs)
        loss_history = self._build_loss_history_from_training_metrics(metrics)
        report_path = self.report([], random_history=None, loss_history=loss_history)

        enriched = frame.copy()
        enriched["active_learning_score"] = np.nan
        enriched["active_learning_rank"] = np.nan
        enriched["active_learning_selected"] = False
        enriched["active_learning_target_column"] = selection.target_column
        enriched["active_learning_feature_columns"] = ", ".join(selection.feature_columns)

        query_rows = pd.DataFrame(columns=enriched.columns)
        if not pool.empty:
            query_rows = self.query(pool, strategy=self.query_strategy, selection=selection)
            scores = query_rows.set_index("__row_index")["active_learning_score"].to_dict()
            ranks = query_rows.set_index("__row_index")["active_learning_rank"].to_dict()
            selected_indices = set(int(value) for value in query_rows["__row_index"].tolist())
            for row_index, score in scores.items():
                enriched.at[row_index, "active_learning_score"] = float(score)
            for row_index, rank in ranks.items():
                enriched.at[row_index, "active_learning_rank"] = int(rank)
            for row_index in selected_indices:
                enriched.at[row_index, "active_learning_selected"] = True

        summary = {
            "task_prompt": selection.task_prompt,
            "target_column": selection.target_column,
            "feature_columns": selection.feature_columns,
            "strategy": self.query_strategy,
            "labeled_rows": int(len(labeled)),
            "pool_rows": int(len(pool)),
            "selected_rows": 0 if query_rows.empty else int(len(query_rows)),
            "classes": sorted({_normalize_label_value(value) for value in labeled[selection.target_column].dropna().tolist()}),
            "history": [
                {
                    "epoch": int(item["epoch"]),
                    "train_loss": float(item["train_loss"]),
                    "val_loss": float(item["val_loss"]),
                }
                for item in loss_history
            ],
            "metrics": metrics,
        }
        artifacts = self._write_artifacts(enriched, query_rows, summary, report_path, prepared)

        self._record_log(f"Wrote active learning dataset to {artifacts.dataset_path}.", logs)
        self._record_log(f"Wrote active learning queries to {artifacts.queries_path}.", logs)
        self._record_log(f"Wrote active learning summary to {artifacts.summary_path}.", logs)
        self._record_log(f"Wrote active learning report to {artifacts.report_path}.", logs)
        self._record_log("ActiveLearningAgent execution finished successfully.", logs)

        schema = {column: str(dtype) for column, dtype in enriched.dtypes.items()}
        upstream_artifacts = dict(upstream.artifacts) if upstream is not None else {}
        upstream_metadata = deepcopy(upstream.metadata) if upstream is not None else {}
        upstream_logs = list(upstream.logs) if upstream is not None else []
        upstream_metrics = dict(upstream.metrics) if upstream is not None else {}
        merged_artifacts = {
            **upstream_artifacts,
            "active_learning_dataset": str(artifacts.dataset_path),
            "active_learning_queries": str(artifacts.queries_path),
            "active_learning_summary": str(artifacts.summary_path),
            "active_learning_report": str(artifacts.report_path),
            "active_learning_train_script": str(artifacts.train_script_path),
            "active_learning_train_dataset": str(artifacts.train_dataset_path),
            "active_learning_val_dataset": str(artifacts.val_dataset_path),
            "active_learning_pool_dataset": str(artifacts.pool_dataset_path),
            "active_learning_model": str(artifacts.model_path),
            "active_learning_training_metrics": str(artifacts.training_metrics_path),
        }
        merged_metadata = {**upstream_metadata, "active_learning": summary}
        merged_metrics = {
            **upstream_metrics,
            "active_learning_accuracy": metrics.get("accuracy"),
            "active_learning_macro_f1": metrics.get("macro_f1"),
            "active_learning_selected_rows": summary["selected_rows"],
        }
        merged_logs = self._merge_logs(upstream_logs, logs)

        return AgentResult(
            dataframe=enriched,
            dataframe_path=artifacts.dataset_path,
            dataframe_schema=schema,
            metrics=merged_metrics,
            artifacts=merged_artifacts,
            logs=merged_logs,
            metadata=merged_metadata,
        )

    def fit(self, labeled_df: pd.DataFrame, *, selection: TaskSelection):
        raise RuntimeError("ActiveLearningAgent is orchestrator-only. Training is executed by CodeAgent in execute().")

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        *,
        selection: TaskSelection,
        model: Any | None = None,
    ):
        raise RuntimeError("ActiveLearningAgent is orchestrator-only. Evaluation comes from training metrics.")

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = DEFAULT_QUERY_STRATEGY,
        *,
        selection: TaskSelection,
    ) -> pd.DataFrame:
        working = pool_df.copy()
        if working.empty:
            return working
        working["__row_index"] = working.index.astype(int)
        texts = self._compose_texts(working, selection.feature_columns, task_prompt=selection.task_prompt)
        scores = self._deterministic_scores(texts, strategy=strategy)
        working["active_learning_score"] = scores
        working = working.sort_values(
            by=["active_learning_score", "__row_index"],
            ascending=[False, True],
            kind="stable",
        ).copy()
        working["active_learning_rank"] = np.arange(1, len(working) + 1)
        working["active_learning_selected"] = working["active_learning_rank"] <= self.batch_size
        return working.loc[working["active_learning_selected"]].copy()

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        *,
        strategy: str = DEFAULT_QUERY_STRATEGY,
        n_iterations: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        task_prompt: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        prompt = self._normalize_text(task_prompt) or self.task_prompt or "Active learning task"
        seed_selection = self._select_task_columns(
            pd.concat([labeled_df, pool_df], axis=0, ignore_index=False),
            {"task_prompt": prompt},
        )
        working_labeled = labeled_df.copy()
        working_pool = pool_df.copy()
        original_batch_size = self.batch_size
        self.batch_size = int(batch_size)
        history: list[dict[str, float | int | str]] = []
        try:
            for iteration in range(1, n_iterations + 1):
                history.append(
                    {
                        "iteration": iteration,
                        "n_labeled": int(len(working_labeled)),
                        "accuracy": 0.0,
                        "macro_f1": 0.0,
                        "strategy": strategy,
                    }
                )
                if working_pool.empty:
                    break
                queried = self.query(working_pool, strategy=strategy, selection=seed_selection)
                if queried.empty:
                    break
                acquired = queried.drop(
                    columns=["active_learning_score", "active_learning_rank", "active_learning_selected", "__row_index"],
                    errors="ignore",
                )
                working_labeled = pd.concat([working_labeled, acquired], ignore_index=False)
                working_pool = working_pool.drop(index=queried["__row_index"].tolist(), errors="ignore")
        finally:
            self.batch_size = original_batch_size
        return history

    def report(
        self,
        history: Sequence[Mapping[str, Any]],
        random_history: Sequence[Mapping[str, Any]] | None = None,
        *,
        loss_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        report_path = self.output_dir / str(self.active_config.get("report_path", DEFAULT_REPORT_PATH))
        plt.figure(figsize=(8, 5))
        ordered = sorted(
            list(loss_history or []),
            key=lambda row: int(row.get("epoch", 0)),
        )
        if ordered:
            plt.plot(
                [int(row["epoch"]) for row in ordered],
                [float(row["train_loss"]) for row in ordered],
                marker="o",
                label="train_loss",
            )
            plt.plot(
                [int(row["epoch"]) for row in ordered],
                [float(row["val_loss"]) for row in ordered],
                marker="o",
                label="val_loss",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No loss history", ha="center", va="center")
            plt.axis("off")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_path, dpi=160)
        plt.close()
        return report_path

    @staticmethod
    def _build_loss_history_from_training_metrics(metrics: Mapping[str, Any]) -> list[dict[str, float | int]]:
        raw = metrics.get("loss_history")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise RuntimeError("Training metrics must include `loss_history` as a list.")
        parsed: list[dict[str, float | int]] = []
        for index, row in enumerate(raw, start=1):
            if not isinstance(row, Mapping):
                continue
            train_loss = row.get("train_loss")
            val_loss = row.get("val_loss")
            if train_loss is None or val_loss is None:
                continue
            parsed.append(
                {
                    "epoch": int(row.get("epoch", index)),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )
        if not parsed:
            raise RuntimeError("Training metrics `loss_history` is empty or malformed.")
        return parsed

    def _select_task_columns(self, frame: pd.DataFrame, payload: Any | None) -> TaskSelection:
        prompt = self._resolve_task_prompt(payload)
        if not prompt:
            raise ValueError("ActiveLearningAgent requires `task_prompt` in config or payload.")
        feature_override = self.active_config.get("feature_columns")
        target_override = self.active_config.get("target_column")
        if isinstance(feature_override, Sequence) and not isinstance(feature_override, (str, bytes)):
            feature_columns = [str(column) for column in feature_override if str(column) in frame.columns]
        else:
            feature_columns = self._infer_feature_columns(frame, prompt)
        if not feature_columns:
            raise ValueError("ActiveLearningAgent could not identify text feature columns for the task prompt.")
        if isinstance(target_override, str):
            if target_override in frame.columns:
                target_column = target_override
            else:
                raise ValueError(
                    f"ActiveLearningAgent target_column {target_override!r} was configured but is missing from the dataframe. "
                    "Check that the upstream annotation stage actually created this field."
                )
        else:
            if DEFAULT_ANNOTATION_TARGET_COLUMN in frame.columns:
                target_column = DEFAULT_ANNOTATION_TARGET_COLUMN
            else:
                raise ValueError(
                    "ActiveLearningAgent requires `annotation_label` from the annotation stage. "
                    "Configure `agents.active_learning.target_column` explicitly only if you intentionally override it."
                )
        return TaskSelection(feature_columns=feature_columns, target_column=target_column, task_prompt=prompt)

    def _infer_feature_columns(self, frame: pd.DataFrame, prompt: str) -> list[str]:
        prompt_tokens = set(TOKEN_PATTERN.findall(prompt.lower()))
        scores: list[tuple[float, str]] = []
        for column in frame.columns:
            series = frame[column]
            if not self._is_text_like(series):
                continue
            name_tokens = set(TOKEN_PATTERN.findall(column.lower()))
            avg_length = float(series.dropna().astype(str).map(len).mean() or 0.0)
            score = 0.0
            score += 3.0 * len(prompt_tokens & name_tokens)
            score += 4.0 if column.lower() in TEXT_COLUMN_HINTS else 0.0
            score += min(avg_length / 80.0, 4.0)
            scores.append((score, column))
        scores.sort(reverse=True)
        if not scores:
            return []
        best_score = scores[0][0]
        return [column for score, column in scores if score >= max(3.0, best_score - 1.5)][:3]

    def _prepare_training_files(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        selection: TaskSelection,
    ) -> ActiveLearningArtifacts:
        train_dataset_path = self.output_dir / "train.jsonl"
        val_dataset_path = self.output_dir / "val.jsonl"
        pool_dataset_path = self.output_dir / "pool.jsonl"
        train_script_path = self.output_dir / str(self.active_config.get("train_script_path", DEFAULT_SCRIPT_PATH))
        model_path = self.output_dir / str(self.active_config.get("model_path", DEFAULT_MODEL_PATH))
        training_metrics_path = self.output_dir / str(
            self.active_config.get("training_metrics_path", DEFAULT_TRAINING_METRICS_PATH)
        )

        train_df.to_json(train_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        val_df.to_json(val_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        pool_df.to_json(pool_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")

        train_script_path.write_text(self._render_training_script(), encoding="utf-8")

        training_payload = {
            "train_path": str(train_dataset_path.resolve()),
            "val_path": str(val_dataset_path.resolve()),
            "feature_columns": selection.feature_columns,
            "target_column": selection.target_column,
            "task_prompt": selection.task_prompt,
            "random_seed": self.random_seed,
            "model_path": str(model_path.resolve()),
            "metrics_path": str(training_metrics_path.resolve()),
        }
        training_config_path = self.output_dir / "training_job_config.json"
        training_config_path.write_text(json.dumps(training_payload, indent=2), encoding="utf-8")

        return ActiveLearningArtifacts(
            dataset_path=self.output_dir / "active_learning_dataset.jsonl",
            queries_path=self.output_dir / "active_learning_queries.jsonl",
            summary_path=self.output_dir / "active_learning_summary.json",
            report_path=self.output_dir / str(self.active_config.get("report_path", DEFAULT_REPORT_PATH)),
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            pool_dataset_path=pool_dataset_path,
            train_script_path=train_script_path,
            model_path=model_path,
            training_metrics_path=training_metrics_path,
        )

    def _render_training_script(self) -> str:
        return (
            "#!/usr/bin/env python3\n"
            '"""Training script placeholder generated by ActiveLearningAgent orchestrator.\n'
            "CodeAgent overwrites and executes this script.\"\"\"\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(0)\n"
        )

    def _run_generated_training(self, artifacts: ActiveLearningArtifacts, logs: list[str]) -> dict[str, Any]:
        self._run_training_locally_with_code_agent(artifacts=artifacts)

        if not artifacts.training_metrics_path.exists():
            raise RuntimeError("Generated training script did not produce training metrics.")
        if not artifacts.model_path.exists():
            raise RuntimeError("Generated training script did not produce model checkpoint.")

        metrics = json.loads(artifacts.training_metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, Mapping):
            raise RuntimeError("Training metrics file must contain a JSON object.")
        self._build_loss_history_from_training_metrics(metrics)

        self._record_log(
            f"Generated training script executed via local CodeAgent: {artifacts.train_script_path}",
            logs,
        )
        return dict(metrics)

    def _run_training_locally_with_code_agent(self, *, artifacts: ActiveLearningArtifacts) -> None:
        try:
            import httpx
            from smolagents import CodeAgent, OpenAIModel
            from smolagents.agents import RunResult
            from agents.tools import build_search_tools
        except Exception as error:
            raise RuntimeError(
                "Agentic local training requires smolagents with OpenAI-compatible model support."
            ) from error

        llm_config = self.config.get("llm", {})
        if not isinstance(llm_config, Mapping):
            llm_config = {}
        base_url = str(
            llm_config.get("api_base")
            or llm_config.get("base_url")
            or "http://localhost:11434/v1"
        )
        api_base = self._normalize_api_base(base_url)
        api_key = str(llm_config.get("api_key") or os.getenv("OPENAI_API_KEY") or "ollama")
        model_id = str(llm_config.get("model", "kimi-k2.5:cloud"))
        model = OpenAIModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            client_kwargs={"http_client": httpx.Client(trust_env=False)},
            temperature=float(llm_config.get("temperature", 0.1)),
            max_tokens=int(llm_config.get("max_tokens", 4000)),
        )
        config_path = self.output_dir / "training_job_config.json"
        task_prompt = self._resolve_task_prompt(None) or ""

        task = self._build_agentic_training_task(
            script_path=str(artifacts.train_script_path.resolve()),
            config_path=str(config_path.resolve()),
            task_prompt=task_prompt,
        )
        tools = build_search_tools()
        instructions = (
            "You are a model-training coding agent. "
            "Write robust Python code, run it, inspect runtime outputs, and refine until training succeeds. "
            "You must produce the requested files and end with strict JSON in final_answer."
        )
        max_steps = max(1, int(self.agent_max_steps))
        with CodeAgent(
            tools=tools,
            model=model,
            executor_type="local",
            executor_kwargs={
                "additional_functions": {
                    "super": super,
                }
            },
            additional_authorized_imports=[
                "json",
                "pathlib",
                "numpy",
                "pandas",
                "sklearn",
                "sklearn.metrics",
                "sklearn.metrics.*",
                "torch",
                "torch.*",
                "tqdm",
                "numpy.*",
                "transformers",
                "transformers.*",
            ],
            max_steps=max_steps,
            verbosity_level=1,
            instructions=instructions,
        ) as agent:
            run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
            if not isinstance(run_result, RunResult):
                raise RuntimeError("Agentic local training returned an unexpected result type.")
            output_payload = self._parse_agent_output(run_result.output)
            if output_payload is not None:
                if output_payload.get("success") is False:
                    raise RuntimeError(f"Agentic local training reported failure: {output_payload}")
                required_keys = {"success", "model_path", "metrics_path", "history"}
                missing = [key for key in required_keys if key not in output_payload]
                if missing:
                    raise RuntimeError(
                        "Agentic local training final JSON is missing required keys: "
                        + ", ".join(sorted(missing))
                    )
                notes_text = str(output_payload.get("notes", "")).lower()

    def _build_agentic_training_task(
        self,
        *,
        script_path: str,
        config_path: str,
        task_prompt: str,
    ) -> str:
        task_line = task_prompt if task_prompt else "No explicit task prompt provided."
        return (
            "Train a classifier using the provided split files and config.\n"
            f"Config JSON path: {config_path}\n"
            f"Output script path to create/update: {script_path}\n"
            f"Primary user task_prompt (highest priority): {task_line}\n"
            "Requirements:\n"
            "1) Read config JSON and use train/val paths exactly as provided.\n"
            "2) Keep training deterministic with explicit random seeds.\n"
            "5) Write model artifact to `model_path` and metrics JSON to `metrics_path` from config.\n"
            "6) Metrics JSON MUST include: accuracy, macro_f1, evaluated_rows, train_rows, val_rows, backend, loss_history.\n"
            "7) `loss_history` MUST be a per-epoch list of {epoch, train_loss, val_loss}.\n"
            "8) Execute the script and verify output files exist before final answer.\n"
            "9) Return final JSON with keys: success (bool), model_path, metrics_path, history, notes."
        )

    @staticmethod
    def _parse_agent_output(raw_output: Any) -> dict[str, Any] | None:
        if raw_output is None:
            return None
        if isinstance(raw_output, Mapping):
            return dict(raw_output)
        text = str(raw_output).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return dict(parsed)
        return None

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        normalized = api_base.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized[: -len("/chat/completions")]
        return normalized

    def _resolve_dataframe(self, payload: Any | None) -> pd.DataFrame:
        if isinstance(payload, AgentResult):
            if payload.dataframe is not None:
                return payload.dataframe.copy()
            if payload.dataframe_path is not None:
                return self._read_dataframe(Path(payload.dataframe_path))
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, Mapping) and isinstance(payload.get("dataframe"), pd.DataFrame):
            return payload["dataframe"].copy()
        input_path = self.active_config.get("input_path")
        if isinstance(input_path, str) and input_path.strip():
            return self._read_dataframe(Path(input_path))
        for fallback in (
            self._stage_output_dir(self.base_output_dir, "annotation") / "annotated_dataset.jsonl",
            self._stage_output_dir(self.base_output_dir, "quality") / "cleaned_dataset.jsonl",
            self._stage_output_dir(self.base_output_dir, "collection") / "unified_dataset.jsonl",
        ):
            if fallback.exists():
                return self._read_dataframe(fallback)
        raise ValueError("ActiveLearningAgent requires a dataframe payload or active_learning.input_path.")

    def _resolve_task_prompt(self, payload: Any | None) -> str | None:
        if isinstance(payload, Mapping):
            prompt = self._normalize_text(payload.get("task_prompt"))
            if prompt:
                return prompt
        return self.task_prompt

    def _resolve_active_learning_config(self) -> dict[str, Any]:
        agents_config = self.config.get("agents", {})
        if isinstance(agents_config, Mapping) and isinstance(agents_config.get("active_learning"), Mapping):
            return dict(agents_config["active_learning"])
        return {}

    def _write_artifacts(
        self,
        enriched: pd.DataFrame,
        queries: pd.DataFrame,
        summary: Mapping[str, Any],
        report_path: Path,
        prepared: ActiveLearningArtifacts,
    ) -> ActiveLearningArtifacts:
        prepared.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_json(prepared.dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        queries.drop(columns=["__row_index"], errors="ignore").to_json(
            prepared.queries_path,
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )
        prepared.summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
        prepared.report_path = report_path
        return prepared

    def _split_train_test(self, frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(frame) < 6:
            return frame.copy(), frame.iloc[0:0].copy()
        shuffled = frame.sample(frac=1.0, random_state=self.random_seed)
        label_series = shuffled[target_column].apply(_normalize_label_value)
        class_groups: dict[str, list[int]] = {}
        for index, label in zip(shuffled.index.tolist(), label_series.tolist(), strict=False):
            class_groups.setdefault(label, []).append(index)

        if len(class_groups) < 2:
            raise ValueError(f"ActiveLearningAgent requires at least two classes in `{target_column}` for splitting.")

        desired_val = int(round(len(shuffled) * self.test_size))
        desired_val = max(1, min(len(shuffled) - len(class_groups), desired_val))
        max_val_total = sum(max(0, len(indices) - 1) for indices in class_groups.values())
        if max_val_total <= 0:
            return shuffled.copy(), shuffled.iloc[0:0].copy()
        desired_val = min(desired_val, max_val_total)

        targets: list[tuple[float, str]] = []
        val_counts = {label: 0 for label in class_groups}
        for label, indices in class_groups.items():
            cap = max(0, len(indices) - 1)
            proportional = (len(indices) / len(shuffled)) * desired_val
            base = min(cap, int(proportional))
            val_counts[label] = base
            fractional = proportional - int(proportional)
            targets.append((fractional, label))

        assigned = sum(val_counts.values())
        for _, label in sorted(targets, reverse=True):
            if assigned >= desired_val:
                break
            cap = max(0, len(class_groups[label]) - 1)
            if val_counts[label] < cap:
                val_counts[label] += 1
                assigned += 1

        val_indices: list[int] = []
        for label, indices in class_groups.items():
            count = val_counts[label]
            if count > 0:
                val_indices.extend(indices[:count])

        val_df = shuffled.loc[val_indices].copy()
        train_df = shuffled.drop(index=val_indices).copy()
        train_unique = train_df[target_column].dropna().apply(_normalize_label_value).nunique()
        if train_unique < 2:
            return shuffled.copy(), shuffled.iloc[0:0].copy()
        return train_df, val_df

    def _compose_texts(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        task_prompt: str | None = None,
    ) -> list[str]:
        return _compose_texts_from_frame(frame, feature_columns, task_prompt=task_prompt)

    def _deterministic_scores(self, texts: Sequence[str], strategy: str) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float32)
        if strategy == "random":
            rng = np.random.default_rng(self.random_seed)
            return rng.random(len(texts)).astype(np.float32)
        if strategy == "margin":
            lengths = np.asarray([len(text) for text in texts], dtype=np.float32)
            if len(lengths) == 0:
                return np.asarray([], dtype=np.float32)
            center = float(np.median(lengths))
            return 1.0 / (1.0 + np.abs(lengths - center))
        scores: list[float] = []
        for text in texts:
            tokens = TOKEN_PATTERN.findall((text or "").lower())
            if not tokens:
                scores.append(0.0)
                continue
            unique = len(set(tokens))
            scores.append(float(unique / len(tokens)))
        return np.asarray(scores, dtype=np.float32)

    def _record_log(self, message: str, logs: list[str]) -> None:
        logs.append(message)

    def _merge_logs(self, upstream_logs: Sequence[str], logs: Sequence[str]) -> list[str]:
        return [*upstream_logs, *logs]

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_json(path, lines=True)

    def _is_text_like(self, series: pd.Series) -> bool:
        if series.dtype.kind not in {"O", "U", "S"} and not pd.api.types.is_string_dtype(series):
            return False
        sample = series.dropna().astype(str).head(10)
        if sample.empty:
            return False
        return float(sample.map(len).mean()) >= 8.0

    def _load_config(self, config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file was not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            return {}
        if not isinstance(loaded, Mapping):
            raise ValueError("ActiveLearningAgent configuration must be a mapping.")
        return dict(loaded)

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _stage_output_dir(base_output_dir: Path, stage: str) -> Path:
        return base_output_dir / stage
