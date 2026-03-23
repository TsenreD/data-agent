import json
import math
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from hashlib import blake2b
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
TARGET_COLUMN_HINTS = {
    "label",
    "target",
    "class",
    "category",
    "sentiment",
    "answer",
    "complete",
    "problem",
}
DEFAULT_QUERY_STRATEGY = "entropy"
DEFAULT_BATCH_SIZE = 20
DEFAULT_REPORT_PATH = "active_learning_curve.png"


@dataclass(slots=True)
class ActiveLearningArtifacts:
    dataset_path: Path
    queries_path: Path
    summary_path: Path
    report_path: Path


@dataclass(slots=True)
class TaskSelection:
    feature_columns: list[str]
    target_column: str
    task_prompt: str


class FastTextClassificationHead:
    def __init__(
        self,
        *,
        dim: int = 64,
        bucket_size: int = 8192,
        min_n: int = 3,
        max_n: int = 6,
        learning_rate: float = 0.05,
        epochs: int = 25,
        seed: int = 13,
    ) -> None:
        self.dim = int(dim)
        self.bucket_size = int(bucket_size)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.embeddings: np.ndarray | None = None
        self.head: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: list[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "FastTextClassificationHead":
        unique_labels = sorted({str(label) for label in labels})
        if len(unique_labels) < 2:
            raise ValueError("Active learning requires at least two label classes.")
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = unique_labels
        self.embeddings = self.rng.normal(0.0, 0.1, size=(self.bucket_size, self.dim))
        self.head = self.rng.normal(0.0, 0.1, size=(self.dim, len(unique_labels)))
        self.bias = np.zeros(len(unique_labels), dtype=float)
        encoded = [self._feature_ids(text) for text in texts]
        y = np.array([self.label_to_index[str(label)] for label in labels], dtype=int)

        indices = np.arange(len(encoded))
        for _ in range(self.epochs):
            self.rng.shuffle(indices)
            for row_index in indices:
                feature_ids = encoded[row_index]
                if not feature_ids:
                    continue
                pooled = self.embeddings[feature_ids].mean(axis=0)
                logits = pooled @ self.head + self.bias
                probabilities = self._softmax(logits)
                probabilities[y[row_index]] -= 1.0
                head_snapshot = self.head.copy()
                self.head -= self.learning_rate * np.outer(pooled, probabilities)
                self.bias -= self.learning_rate * probabilities
                grad_embedding = head_snapshot @ probabilities
                unique_ids, counts = np.unique(feature_ids, return_counts=True)
                scale = counts.astype(float) / float(len(feature_ids))
                self.embeddings[unique_ids] -= self.learning_rate * np.outer(scale, grad_embedding)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.embeddings is None or self.head is None or self.bias is None:
            raise ValueError("Model must be fit before prediction.")
        rows: list[np.ndarray] = []
        uniform = np.full(len(self.index_to_label), 1.0 / len(self.index_to_label), dtype=float)
        for text in texts:
            feature_ids = self._feature_ids(text)
            if not feature_ids:
                rows.append(uniform.copy())
                continue
            pooled = self.embeddings[feature_ids].mean(axis=0)
            rows.append(self._softmax(pooled @ self.head + self.bias))
        return np.vstack(rows)

    def predict(self, texts: Sequence[str]) -> list[str]:
        probabilities = self.predict_proba(texts)
        return [self.index_to_label[int(np.argmax(row))] for row in probabilities]

    def _feature_ids(self, text: str) -> list[int]:
        tokens = TOKEN_PATTERN.findall((text or "").lower())
        feature_ids: list[int] = []
        for token in tokens:
            feature_ids.append(self._hash(token))
            wrapped = f"<{token}>"
            for size in range(self.min_n, self.max_n + 1):
                if len(wrapped) < size:
                    continue
                for start in range(0, len(wrapped) - size + 1):
                    feature_ids.append(self._hash(wrapped[start : start + size]))
        return feature_ids

    def _hash(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.bucket_size

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        exp = np.exp(shifted)
        return exp / exp.sum()


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
        self.model_config = self._resolve_model_config()
        self._model: FastTextClassificationHead | None = None

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

        model, metrics = self.fit(labeled, selection=selection)
        self._model = model
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

        learning_curves = self._build_learning_curve(labeled, selection)
        report_path = self.report(
            learning_curves["strategy_history"],
            random_history=learning_curves["random_history"],
        )
        summary = {
            "task_prompt": selection.task_prompt,
            "target_column": selection.target_column,
            "feature_columns": selection.feature_columns,
            "strategy": self.query_strategy,
            "labeled_rows": int(len(labeled)),
            "pool_rows": int(len(pool)),
            "selected_rows": 0 if query_rows.empty else int(len(query_rows)),
            "classes": sorted({str(value) for value in labeled[selection.target_column].dropna().tolist()}),
            "history": learning_curves["strategy_history"],
            "random_history": learning_curves["random_history"],
            "metrics": metrics,
        }
        artifacts = self._write_artifacts(enriched, query_rows, summary, report_path)

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

    def fit(
        self,
        labeled_df: pd.DataFrame,
        *,
        selection: TaskSelection,
    ) -> tuple[FastTextClassificationHead, dict[str, float | int]]:
        train_df, test_df = self._split_train_test(labeled_df, selection.target_column)
        model = FastTextClassificationHead(**self.model_config)
        model.fit(self._compose_texts(train_df, selection.feature_columns), self._target_labels(train_df, selection.target_column))
        metrics = self.evaluate(train_df, test_df, selection=selection, model=model)
        return model, metrics

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = DEFAULT_QUERY_STRATEGY,
        *,
        selection: TaskSelection,
    ) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model must be fit before querying.")
        working = pool_df.copy()
        working["__row_index"] = working.index.astype(int)
        texts = self._compose_texts(working, selection.feature_columns)
        probabilities = self._model.predict_proba(texts)
        working["active_learning_score"] = self._uncertainty_scores(probabilities, strategy)
        working = working.sort_values("active_learning_score", ascending=False, kind="stable").copy()
        working["active_learning_rank"] = np.arange(1, len(working) + 1)
        working["active_learning_selected"] = working["active_learning_rank"] <= self.batch_size
        return working.loc[working["active_learning_selected"]].copy()

    def evaluate(
        self,
        labeled_df: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        *,
        selection: TaskSelection,
        model: FastTextClassificationHead | None = None,
    ) -> dict[str, float | int]:
        estimator = model or self._model
        if estimator is None:
            raise ValueError("Model must be fit before evaluation.")
        evaluation_frame = labeled_df if test_df is None or test_df.empty else test_df
        truth = self._target_labels(evaluation_frame, selection.target_column)
        predicted = estimator.predict(self._compose_texts(evaluation_frame, selection.feature_columns))
        return {
            "accuracy": self._accuracy(truth, predicted),
            "macro_f1": self._macro_f1(truth, predicted),
            "evaluated_rows": int(len(evaluation_frame)),
            "train_rows": int(len(labeled_df)),
        }

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
                model, metrics = self.fit(working_labeled, selection=seed_selection)
                self._model = model
                history.append(
                    {
                        "iteration": iteration,
                        "n_labeled": int(len(working_labeled)),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "strategy": strategy,
                    }
                )
                if working_pool.empty:
                    break
                queried = self.query(working_pool, strategy=strategy, selection=seed_selection)
                if queried.empty:
                    break
                acquired = queried.drop(columns=["active_learning_score", "active_learning_rank", "active_learning_selected"], errors="ignore")
                working_labeled = pd.concat([working_labeled, acquired], ignore_index=False)
                working_pool = working_pool.drop(index=queried["__row_index"].tolist(), errors="ignore")
                working_pool = working_pool.drop(columns=["__row_index"], errors="ignore")
        finally:
            self.batch_size = original_batch_size
        return history

    def report(
        self,
        history: Sequence[Mapping[str, Any]],
        random_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        report_path = self.output_dir / str(self.active_config.get("report_path", DEFAULT_REPORT_PATH))
        plt.figure(figsize=(8, 5))
        if history:
            plt.plot(
                [int(row["n_labeled"]) for row in history],
                [float(row["macro_f1"]) for row in history],
                marker="o",
                label=str(history[0].get("strategy", self.query_strategy)),
            )
        if random_history:
            plt.plot(
                [int(row["n_labeled"]) for row in random_history],
                [float(row["macro_f1"]) for row in random_history],
                marker="o",
                label=str(random_history[0].get("strategy", "random")),
            )
        plt.xlabel("Labeled examples")
        plt.ylabel("Macro F1")
        plt.title("Active Learning Curve")
        plt.grid(True, alpha=0.3)
        if history or random_history:
            plt.legend()
        plt.tight_layout()
        plt.savefig(report_path, dpi=160)
        plt.close()
        return report_path

    def _build_learning_curve(self, labeled_df: pd.DataFrame, selection: TaskSelection) -> dict[str, list[dict[str, float | int | str]]]:
        if len(labeled_df) < 6:
            return {"strategy_history": [], "random_history": []}
        shuffled = labeled_df.sample(frac=1.0, random_state=self.random_seed)
        baseline_size = max(4, min(len(shuffled) - 2, self.batch_size))
        initial = shuffled.iloc[:baseline_size].copy()
        pool = shuffled.iloc[baseline_size:].copy()
        if pool.empty:
            return {"strategy_history": [], "random_history": []}
        iterations = max(1, min(4, math.ceil(len(pool) / max(1, self.batch_size))))
        prompt = selection.task_prompt
        history = self.run_cycle(
            initial,
            pool,
            strategy=self.query_strategy,
            n_iterations=iterations,
            batch_size=min(self.batch_size, max(1, len(pool))),
            task_prompt=prompt,
        )
        if not history:
            return {"strategy_history": [], "random_history": []}
        random_history = self.run_cycle(
            initial,
            pool,
            strategy="random",
            n_iterations=iterations,
            batch_size=min(self.batch_size, max(1, len(pool))),
            task_prompt=prompt,
        )
        return {"strategy_history": history, "random_history": random_history}

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
            target_column = self._infer_target_column(frame, prompt)
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
            if "text" in prompt_tokens and "text" in name_tokens:
                score += 5.0
            scores.append((score, column))
        scores.sort(reverse=True)
        if not scores:
            return []
        best_score = scores[0][0]
        return [column for score, column in scores if score >= max(3.0, best_score - 1.5)][:3]

    def _infer_target_column(self, frame: pd.DataFrame, prompt: str) -> str:
        prompt_tokens = set(TOKEN_PATTERN.findall(prompt.lower()))
        best_column: str | None = None
        best_score = float("-inf")
        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            if non_null.empty:
                continue
            if self._is_text_like(series):
                continue
            unique_values = {str(value).lower() for value in non_null.tolist()}
            if len(unique_values) < 2:
                continue
            if len(unique_values) > max(20, int(len(non_null) * 0.6)):
                continue
            name_tokens = set(TOKEN_PATTERN.findall(column.lower()))
            value_tokens = set()
            for value in list(unique_values)[:10]:
                value_tokens.update(TOKEN_PATTERN.findall(value))
            score = 0.0
            score += 4.0 * len(prompt_tokens & name_tokens)
            score += 2.0 * len(prompt_tokens & value_tokens)
            score += 5.0 if column.lower() in TARGET_COLUMN_HINTS else 0.0
            if {"full", "incomplete", "complete"} & prompt_tokens and {"complete", "incomplete"} & name_tokens:
                score += 6.0
            if len(unique_values) <= 2:
                score += 4.0
            missing_share = 1.0 - (len(non_null) / max(1, len(series)))
            score += 2.0 * missing_share
            if score > best_score:
                best_score = score
                best_column = column
        if best_column is None:
            raise ValueError("ActiveLearningAgent could not identify a target column from the task prompt.")
        return best_column

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

    def _resolve_model_config(self) -> dict[str, Any]:
        raw = self.active_config.get("model", {})
        config = dict(raw) if isinstance(raw, Mapping) else {}
        return {
            "dim": int(config.get("dim", 64)),
            "bucket_size": int(config.get("bucket_size", 8192)),
            "min_n": int(config.get("min_n", 3)),
            "max_n": int(config.get("max_n", 6)),
            "learning_rate": float(config.get("learning_rate", 0.05)),
            "epochs": int(config.get("epochs", 25)),
            "seed": self.random_seed,
        }

    def _write_artifacts(
        self,
        enriched: pd.DataFrame,
        queries: pd.DataFrame,
        summary: Mapping[str, Any],
        report_path: Path,
    ) -> ActiveLearningArtifacts:
        dataset_path = self.output_dir / "active_learning_dataset.jsonl"
        queries_path = self.output_dir / "active_learning_queries.jsonl"
        summary_path = self.output_dir / "active_learning_summary.json"
        enriched.to_json(dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        queries.drop(columns=["__row_index"], errors="ignore").to_json(
            queries_path,
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )
        summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")
        return ActiveLearningArtifacts(
            dataset_path=dataset_path,
            queries_path=queries_path,
            summary_path=summary_path,
            report_path=report_path,
        )

    def _split_train_test(self, frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(frame) < 6:
            return frame.copy(), frame.iloc[0:0].copy()
        shuffled = frame.sample(frac=1.0, random_state=self.random_seed)
        test_size = max(1, min(len(shuffled) - 2, int(round(len(shuffled) * self.test_size))))
        return shuffled.iloc[:-test_size].copy(), shuffled.iloc[-test_size:].copy()

    def _compose_texts(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
        texts: list[str] = []
        for _, row in frame.iterrows():
            chunks = []
            for column in feature_columns:
                value = row.get(column)
                if pd.isna(value):
                    continue
                chunks.append(str(value))
            texts.append("\n".join(chunks))
        return texts

    def _target_labels(self, frame: pd.DataFrame, target_column: str) -> list[str]:
        return [self._normalize_label(value) for value in frame[target_column].tolist()]

    def _uncertainty_scores(self, probabilities: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == "random":
            rng = np.random.default_rng(self.random_seed)
            return rng.random(len(probabilities))
        if strategy == "margin":
            sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
            return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        epsilon = 1e-12
        entropy = -(probabilities * np.log(probabilities + epsilon)).sum(axis=1)
        return entropy

    def _normalize_label(self, value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        return str(value).strip()

    def _record_log(self, message: str, logs: list[str]) -> None:
        logs.append(message)

    def _merge_logs(self, upstream_logs: Sequence[str], logs: Sequence[str]) -> list[str]:
        return [*upstream_logs, *logs]

    def _accuracy(self, truth: Sequence[str], predicted: Sequence[str]) -> float:
        if not truth:
            return 0.0
        matches = sum(1 for expected, actual in zip(truth, predicted, strict=False) if expected == actual)
        return matches / len(truth)

    def _macro_f1(self, truth: Sequence[str], predicted: Sequence[str]) -> float:
        labels = sorted(set(truth) | set(predicted))
        if not labels:
            return 0.0
        scores: list[float] = []
        for label in labels:
            tp = sum(1 for expected, actual in zip(truth, predicted, strict=False) if expected == label and actual == label)
            fp = sum(1 for expected, actual in zip(truth, predicted, strict=False) if expected != label and actual == label)
            fn = sum(1 for expected, actual in zip(truth, predicted, strict=False) if expected == label and actual != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            if precision + recall == 0.0:
                scores.append(0.0)
            else:
                scores.append(2.0 * precision * recall / (precision + recall))
        return float(sum(scores) / len(scores))

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
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _normalize_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _stage_output_dir(self, output_dir: Path, stage_name: str) -> Path:
        if output_dir.name == stage_name:
            return output_dir
        return output_dir / stage_name
