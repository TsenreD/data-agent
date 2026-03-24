import argparse
import json
import math
import os
import pickle
import random
import re
import runpy
import subprocess
import sys
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


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by environments without torch
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


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
DEFAULT_SCRIPT_PATH = "train_active_learning.py"
DEFAULT_MODEL_PATH = "model.pth"
DEFAULT_TRAINING_METRICS_PATH = "training_metrics.json"
DEFAULT_TRAINING_IMAGE = "data-agent-active-learning-train"
DEFAULT_DOCKER_MEMORY_LIMIT = "16g"
DEFAULT_DOCKER_CPU_LIMIT = 8.0
DEFAULT_DOCKER_PIDS_LIMIT = 2048
DEFAULT_DOCKER_SHM_SIZE = "8g"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DOCKERFILE_PATH = Path(__file__).with_name("Dockerfile.train")


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


@dataclass(slots=True)
class VocabularyEncoder:
    token_to_index: dict[str, int]
    unknown_index: int = 0

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        max_vocab: int,
        min_token_freq: int,
    ) -> "VocabularyEncoder":
        counts: dict[str, int] = {}
        for text in texts:
            for token in TOKEN_PATTERN.findall((text or "").lower()):
                counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        kept = [token for token, freq in ranked if freq >= max(1, int(min_token_freq))][: max(0, int(max_vocab) - 1)]
        token_to_index = {token: index + 1 for index, token in enumerate(kept)}
        return cls(token_to_index=token_to_index, unknown_index=0)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.token_to_index) + 1), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in TOKEN_PATTERN.findall((text or "").lower()):
                token_index = self.token_to_index.get(token, self.unknown_index)
                matrix[row_index, token_index] += 1.0
        return matrix

    def to_payload(self) -> dict[str, Any]:
        return {
            "token_to_index": dict(self.token_to_index),
            "unknown_index": int(self.unknown_index),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "VocabularyEncoder":
        raw = payload.get("token_to_index", {})
        token_to_index = {str(token): int(index) for token, index in dict(raw).items()}
        unknown_index = int(payload.get("unknown_index", 0))
        return cls(token_to_index=token_to_index, unknown_index=unknown_index)


class CheckpointClassifier:
    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        self.checkpoint = dict(checkpoint)
        self.architecture = str(self.checkpoint.get("architecture", "numpy_softmax"))
        self.vectorizer = VocabularyEncoder.from_payload(self.checkpoint["vectorizer"])
        self.index_to_label = [str(item) for item in self.checkpoint["index_to_label"]]

        if self.architecture == "torch_mlp":
            self.w1 = np.asarray(self.checkpoint["w1"], dtype=np.float32)
            self.b1 = np.asarray(self.checkpoint["b1"], dtype=np.float32)
            self.w2 = np.asarray(self.checkpoint["w2"], dtype=np.float32)
            self.b2 = np.asarray(self.checkpoint["b2"], dtype=np.float32)
            self.weights = None
            self.bias = None
        else:
            self.weights = np.asarray(self.checkpoint["weights"], dtype=np.float32)
            self.bias = np.asarray(self.checkpoint["bias"], dtype=np.float32)
            self.w1 = None
            self.b1 = None
            self.w2 = None
            self.b2 = None

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        return self._predict_proba_features(features)

    def predict(self, texts: Sequence[str]) -> list[str]:
        probabilities = self.predict_proba(texts)
        return [self.index_to_label[int(np.argmax(row))] for row in probabilities]

    def _predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros((0, len(self.index_to_label)), dtype=np.float32)
        if self.architecture == "torch_mlp":
            hidden = np.maximum(0.0, features @ self.w1.T + self.b1)
            logits = hidden @ self.w2.T + self.b2
        else:
            logits = features @ self.weights + self.bias
        return _softmax_rows(logits)

    @classmethod
    def load(cls, path: Path) -> "CheckpointClassifier":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("Invalid model checkpoint payload.")
        return cls(payload)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    sums = np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)
    return exp / sums


def _encode_labels(labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        raise ValueError("Active learning requires at least two label classes.")
    label_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([label_to_index[str(label)] for label in labels], dtype=np.int64)
    return classes, encoded


def _accuracy(truth: Sequence[str], predicted: Sequence[str]) -> float:
    if not truth:
        return 0.0
    matches = sum(1 for expected, actual in zip(truth, predicted, strict=False) if expected == actual)
    return float(matches / len(truth))


def _macro_f1(truth: Sequence[str], predicted: Sequence[str]) -> float:
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


def _train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend requested but torch is not installed.")

    if torch is None or nn is None or DataLoader is None or TensorDataset is None:  # pragma: no cover
        raise RuntimeError("Torch modules are unavailable.")

    class _TorchMLP(nn.Module):
        def __init__(self, input_dim: int, inner_dim: int, output_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, inner_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(inner_dim, output_dim)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            hidden = self.relu(self.fc1(features))
            return self.fc2(hidden)

    rng_seed = int(seed)
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)

    x_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)
    effective_batch = max(1, min(int(batch_size), len(dataset)))
    loader = DataLoader(dataset, batch_size=effective_batch, shuffle=True)

    model = _TorchMLP(x_train.shape[1], max(2, int(hidden_dim)), int(len(set(y_train.tolist()))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(max(1, int(epochs))):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    state = model.state_dict()
    w1 = state["fc1.weight"].detach().cpu().numpy().astype(np.float32)
    b1 = state["fc1.bias"].detach().cpu().numpy().astype(np.float32)
    w2 = state["fc2.weight"].detach().cpu().numpy().astype(np.float32)
    b2 = state["fc2.bias"].detach().cpu().numpy().astype(np.float32)
    return w1, b1, w2, b2


def _train_numpy_softmax(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n_features = int(x_train.shape[1])
    n_classes = int(len(set(y_train.tolist())))
    weights = rng.normal(0.0, 0.05, size=(n_features, n_classes)).astype(np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)

    y_one_hot = np.eye(n_classes, dtype=np.float32)[y_train]
    n_rows = float(max(1, len(x_train)))
    for _ in range(max(1, int(epochs))):
        logits = x_train @ weights + bias
        probs = _softmax_rows(logits)
        error = probs - y_one_hot
        grad_w = (x_train.T @ error) / n_rows
        grad_b = error.mean(axis=0)
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
    return weights, bias


def train_text_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    *,
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    training_config: Mapping[str, Any],
    random_seed: int,
) -> tuple[CheckpointClassifier, dict[str, Any], dict[str, Any]]:
    config = dict(training_config)
    vectorizer = VocabularyEncoder.fit(
        train_texts,
        max_vocab=int(config.get("max_vocab", 5000)),
        min_token_freq=int(config.get("min_token_freq", 1)),
    )
    x_train = vectorizer.transform(train_texts)
    classes, y_train = _encode_labels(train_labels)
    label_to_index = {label: index for index, label in enumerate(classes)}

    if val_texts:
        x_val = vectorizer.transform(val_texts)
        y_val = np.asarray([label_to_index[str(label)] for label in val_labels], dtype=np.int64)
        eval_features = x_val
        eval_truth = [str(item) for item in val_labels]
    else:
        eval_features = x_train
        eval_truth = [str(item) for item in train_labels]

    epochs = int(config.get("epochs", 20))
    learning_rate = float(config.get("learning_rate", 0.05))
    hidden_dim = int(config.get("embedding_dim", 64))
    batch_size = int(config.get("batch_size", 32))

    checkpoint: dict[str, Any]
    if TORCH_AVAILABLE:
        w1, b1, w2, b2 = _train_torch_mlp(
            x_train,
            y_train,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=random_seed,
        )
        checkpoint = {
            "architecture": "torch_mlp",
            "vectorizer": vectorizer.to_payload(),
            "index_to_label": classes,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        }
        backend = "torch"
    else:
        weights, bias = _train_numpy_softmax(
            x_train,
            y_train,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=random_seed,
        )
        checkpoint = {
            "architecture": "numpy_softmax",
            "vectorizer": vectorizer.to_payload(),
            "index_to_label": classes,
            "weights": weights,
            "bias": bias,
        }
        backend = "numpy_fallback"

    model = CheckpointClassifier(checkpoint)
    probabilities = model._predict_proba_features(eval_features)
    predicted = [classes[int(np.argmax(row))] for row in probabilities]
    metrics = {
        "accuracy": _accuracy(eval_truth, predicted),
        "macro_f1": _macro_f1(eval_truth, predicted),
        "evaluated_rows": int(len(eval_truth)),
        "train_rows": int(len(train_texts)),
        "val_rows": int(len(val_texts)),
        "backend": backend,
    }
    return model, metrics, checkpoint


def run_training_job(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    train_path = Path(payload["train_path"])
    val_path = Path(payload["val_path"])
    feature_columns = [str(item) for item in payload["feature_columns"]]
    target_column = str(payload["target_column"])

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True) if val_path.exists() else pd.DataFrame(columns=train_df.columns)

    train_texts = _compose_texts_from_frame(train_df, feature_columns)
    train_labels = [_normalize_label_value(item) for item in train_df[target_column].tolist()]
    val_texts = _compose_texts_from_frame(val_df, feature_columns)
    val_labels = [_normalize_label_value(item) for item in val_df[target_column].tolist()] if not val_df.empty else []

    model, metrics, checkpoint = train_text_classifier(
        train_texts,
        train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        training_config=payload.get("training", {}),
        random_seed=int(payload.get("random_seed", 13)),
    )

    model_path = Path(payload["model_path"])
    metrics_path = Path(payload["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _compose_texts_from_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    texts: list[str] = []
    for _, row in frame.iterrows():
        chunks: list[str] = []
        for column in feature_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            chunks.append(str(value))
        texts.append("\n".join(chunks))
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
        self.training_config = self._resolve_training_config()
        self.docker_config = self._resolve_docker_config()
        self._model: CheckpointClassifier | None = None

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

        train_df, val_df = self._split_train_test(labeled, selection.target_column)
        prepared = self._prepare_training_files(train_df, val_df, pool, selection)
        metrics = self._run_generated_training(prepared, logs)

        self._model = CheckpointClassifier.load(prepared.model_path)
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
            "classes": sorted({_normalize_label_value(value) for value in labeled[selection.target_column].dropna().tolist()}),
            "history": learning_curves["strategy_history"],
            "random_history": learning_curves["random_history"],
            "metrics": metrics,
            "training_backend": metrics.get("backend"),
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

    def fit(
        self,
        labeled_df: pd.DataFrame,
        *,
        selection: TaskSelection,
    ) -> tuple[CheckpointClassifier, dict[str, float | int | str]]:
        train_df, val_df = self._split_train_test(labeled_df, selection.target_column)
        train_texts = self._compose_texts(train_df, selection.feature_columns)
        train_labels = self._target_labels(train_df, selection.target_column)
        val_texts = self._compose_texts(val_df, selection.feature_columns)
        val_labels = self._target_labels(val_df, selection.target_column) if not val_df.empty else []
        model, metrics, _ = train_text_classifier(
            train_texts,
            train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            training_config=self.training_config,
            random_seed=self.random_seed,
        )
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
        model: CheckpointClassifier | None = None,
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
            unique_values = {str(value).lower() for value in non_null.tolist()}
            if len(unique_values) < 2:
                continue
            if len(unique_values) > max(40, int(len(non_null) * 0.9)):
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

        script_text = self._render_training_script()
        train_script_path.write_text(script_text, encoding="utf-8")

        training_payload = {
            "train_path": str(train_dataset_path),
            "val_path": str(val_dataset_path),
            "feature_columns": selection.feature_columns,
            "target_column": selection.target_column,
            "training": self.training_config,
            "random_seed": self.random_seed,
            "model_path": str(model_path),
            "metrics_path": str(training_metrics_path),
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
            "\"\"\"Auto-generated by ActiveLearningAgent.\"\"\"\n"
            "import argparse\n"
            "from agents.active_learning.active_learning_agent import run_training_job\n"
            "\n"
            "\n"
            "def main() -> int:\n"
            "    parser = argparse.ArgumentParser(description='Run active-learning training job')\n"
            "    parser.add_argument('--config', required=True, help='Path to training job config JSON')\n"
            "    args = parser.parse_args()\n"
            "    run_training_job(args.config)\n"
            "    return 0\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    def _run_generated_training(self, artifacts: ActiveLearningArtifacts, logs: list[str]) -> dict[str, Any]:
        if self.docker_config["enabled"]:
            self._run_training_in_docker(artifacts)
        else:
            self._run_training_locally(artifacts)

        if not artifacts.training_metrics_path.exists():
            raise RuntimeError("Generated training script did not produce training metrics.")
        if not artifacts.model_path.exists():
            raise RuntimeError("Generated training script did not produce model checkpoint.")

        metrics = json.loads(artifacts.training_metrics_path.read_text(encoding="utf-8"))
        location = "docker" if self.docker_config["enabled"] else "local interpreter"
        self._record_log(f"Generated training script executed via {location}: {artifacts.train_script_path}", logs)
        return metrics

    def _run_training_locally(self, artifacts: ActiveLearningArtifacts) -> None:
        training_config_path = self.output_dir / "training_job_config.json"
        previous_argv = list(sys.argv)
        try:
            sys.argv = [str(artifacts.train_script_path), "--config", str(training_config_path)]
            runpy.run_path(str(artifacts.train_script_path), run_name="__main__")
        except SystemExit as exit_signal:
            code = int(exit_signal.code) if isinstance(exit_signal.code, int) else 1
            if code != 0:
                raise RuntimeError(f"Generated training script failed with exit code {code}.") from exit_signal
        finally:
            sys.argv = previous_argv

    def _run_training_in_docker(self, artifacts: ActiveLearningArtifacts) -> None:
        if not TRAIN_DOCKERFILE_PATH.exists():
            raise RuntimeError(f"Active-learning Dockerfile is missing: {TRAIN_DOCKERFILE_PATH}")

        output_mount_host = self.output_dir.resolve()
        output_mount_container = "/training_io"
        project_mount_host = PROJECT_ROOT.resolve()
        project_mount_container = "/workspace"

        container_config_path = self._prepare_container_training_config(output_mount_container=output_mount_container)

        image_name = str(self.docker_config["image_name"])
        build_new_image = bool(self.docker_config["build_new_image"])
        if not build_new_image and not self._docker_image_exists(image_name):
            build_new_image = True
        if build_new_image:
            self._docker_build_image(image_name)

        if bool(self.docker_config["agentic"]):
            self._run_training_with_code_agent(
                image_name=image_name,
                output_mount_host=output_mount_host,
                output_mount_container=output_mount_container,
                project_mount_host=project_mount_host,
                project_mount_container=project_mount_container,
                script_name=artifacts.train_script_path.name,
                config_name=container_config_path.name,
            )
            return

        run_command = self._build_docker_run_command(
            image_name=image_name,
            output_mount_host=output_mount_host,
            output_mount_container=output_mount_container,
            project_mount_host=project_mount_host,
            project_mount_container=project_mount_container,
            script_name=artifacts.train_script_path.name,
            config_name=container_config_path.name,
        )
        run_result = subprocess.run(run_command, capture_output=True, text=True)
        if run_result.returncode != 0:
            stderr = run_result.stderr.strip()
            stdout = run_result.stdout.strip()
            details = stderr or stdout or "unknown error"
            raise RuntimeError(f"Docker training failed: {details}")

    def _prepare_container_training_config(self, *, output_mount_container: str) -> Path:
        host_config_path = self.output_dir / "training_job_config.json"
        config_payload = json.loads(host_config_path.read_text(encoding="utf-8"))
        config_payload["train_path"] = f"{output_mount_container}/{Path(config_payload['train_path']).name}"
        config_payload["val_path"] = f"{output_mount_container}/{Path(config_payload['val_path']).name}"
        config_payload["model_path"] = f"{output_mount_container}/{Path(config_payload['model_path']).name}"
        config_payload["metrics_path"] = f"{output_mount_container}/{Path(config_payload['metrics_path']).name}"
        container_config_path = self.output_dir / "training_job_config.container.json"
        container_config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        return container_config_path

    def _run_training_with_code_agent(
        self,
        *,
        image_name: str,
        output_mount_host: Path,
        output_mount_container: str,
        project_mount_host: Path,
        project_mount_container: str,
        script_name: str,
        config_name: str,
    ) -> None:
        try:
            import httpx
            from smolagents import CodeAgent, OpenAIModel
            from smolagents.agents import RunResult
            from smolagents.monitoring import AgentLogger, LogLevel
            from agents.data_collection.smolagents_backend import PrebakedDockerExecutor
        except Exception as error:  # pragma: no cover - dependency failure fallback
            raise RuntimeError(
                "Agentic docker training requires smolagents with OpenAI-compatible model support."
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

        sandbox_config = llm_config.get("sandbox", {}) if isinstance(llm_config, Mapping) else {}
        sandbox_mapping = dict(sandbox_config) if isinstance(sandbox_config, Mapping) else {}
        host = str(self.docker_config.get("host", sandbox_mapping.get("host", "127.0.0.1")))
        port = int(self.docker_config.get("port", sandbox_mapping.get("port", 8892)))

        container_run_kwargs = {
            "mem_limit": str(self.docker_config["memory_limit"]),
            "cpu_quota": int(float(self.docker_config["cpu_limit"]) * 100000),
            "pids_limit": int(self.docker_config["pids_limit"]),
            "shm_size": str(self.docker_config["shm_size"]),
            "volumes": {
                str(project_mount_host): {"bind": project_mount_container, "mode": "ro"},
                str(output_mount_host): {"bind": output_mount_container, "mode": "rw"},
            },
            "working_dir": project_mount_container,
            "environment": {"PYTHONPATH": project_mount_container},
        }
        dockerfile_content = TRAIN_DOCKERFILE_PATH.read_text(encoding="utf-8")

        executor = PrebakedDockerExecutor(
            host=host,
            port=port,
            image_name=image_name,
            build_new_image=False,
            container_run_kwargs=container_run_kwargs,
            dockerfile_content=dockerfile_content,
            additional_imports=[
                "json",
                "pathlib",
                "pickle",
                "numpy",
                "pandas",
                "torch",
            ],
            logger=AgentLogger(level=LogLevel.ERROR),
        )

        task = self._build_agentic_training_task(
            output_mount_container=output_mount_container,
            script_name=script_name,
            config_name=config_name,
        )
        instructions = (
            "You are a model-training coding agent. "
            "Write robust Python code, run it, inspect runtime outputs, and refine until training succeeds. "
            "You must produce the requested files and end with strict JSON in final_answer."
        )
        max_steps = int(self.docker_config["max_steps"])
        with CodeAgent(
            tools=[],
            model=model,
            executor=executor,
            executor_type="docker",
            additional_authorized_imports=[
                "json",
                "pathlib",
                "pickle",
                "numpy",
                "pandas",
                "torch",
            ],
            max_steps=max_steps,
            verbosity_level=1,
            instructions=instructions,
        ) as agent:
            run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
            if not isinstance(run_result, RunResult):
                raise RuntimeError("Agentic docker training returned an unexpected result type.")

    def _build_agentic_training_task(
        self,
        *,
        output_mount_container: str,
        script_name: str,
        config_name: str,
    ) -> str:
        script_path = f"{output_mount_container}/{script_name}"
        config_path = f"{output_mount_container}/{config_name}"
        return (
            "Train an active-learning classifier using the provided config and dataset files.\n"
            f"Config JSON path: {config_path}\n"
            f"Output script path to create/update: {script_path}\n"
            "Requirements:\n"
            "1) Read config JSON and train a torch-based classifier.\n"
            "2) Write checkpoint to `model_path` and metrics JSON to `metrics_path` from config.\n"
            "3) If first implementation fails, debug using errors and rerun.\n"
            "4) Keep script deterministic with explicit random seeds.\n"
            "5) Return final JSON with keys: success (bool), model_path, metrics_path, notes.\n"
            "Use only Python code execution and finish with final_answer(json.dumps(...))."
        )

    def _docker_image_exists(self, image_name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "--type=image", image_name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _docker_build_image(self, image_name: str) -> None:
        build_command = [
            "docker",
            "build",
            "-f",
            str(TRAIN_DOCKERFILE_PATH),
            "-t",
            image_name,
            str(PROJECT_ROOT),
        ]
        result = subprocess.run(build_command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "unknown error"
            raise RuntimeError(f"Unable to build Docker image for active learning training: {details}")

    def _build_docker_run_command(
        self,
        *,
        image_name: str,
        output_mount_host: Path,
        output_mount_container: str,
        project_mount_host: Path,
        project_mount_container: str,
        script_name: str,
        config_name: str,
    ) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--memory",
            str(self.docker_config["memory_limit"]),
            "--cpus",
            str(self.docker_config["cpu_limit"]),
            "--pids-limit",
            str(self.docker_config["pids_limit"]),
            "--shm-size",
            str(self.docker_config["shm_size"]),
            "-e",
            f"PYTHONPATH={project_mount_container}",
            "-v",
            f"{project_mount_host}:{project_mount_container}:ro",
            "-v",
            f"{output_mount_host}:{output_mount_container}:rw",
            "-w",
            project_mount_container,
            image_name,
            "python",
            f"{output_mount_container}/{script_name}",
            "--config",
            f"{output_mount_container}/{config_name}",
        ]

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

    def _resolve_training_config(self) -> dict[str, Any]:
        training = self.active_config.get("training", {})
        training_config = dict(training) if isinstance(training, Mapping) else {}

        legacy_model = self.active_config.get("model", {})
        if isinstance(legacy_model, Mapping):
            if "embedding_dim" not in training_config and "dim" in legacy_model:
                training_config["embedding_dim"] = int(legacy_model["dim"])
            if "epochs" not in training_config and "epochs" in legacy_model:
                training_config["epochs"] = int(legacy_model["epochs"])
            if "learning_rate" not in training_config and "learning_rate" in legacy_model:
                training_config["learning_rate"] = float(legacy_model["learning_rate"])

        return {
            "embedding_dim": int(training_config.get("embedding_dim", 64)),
            "epochs": int(training_config.get("epochs", 20)),
            "batch_size": int(training_config.get("batch_size", 32)),
            "learning_rate": float(training_config.get("learning_rate", 0.05)),
            "max_vocab": int(training_config.get("max_vocab", 5000)),
            "min_token_freq": int(training_config.get("min_token_freq", 1)),
        }

    def _resolve_docker_config(self) -> dict[str, Any]:
        docker_config = self.active_config.get("docker", {})
        resolved = dict(docker_config) if isinstance(docker_config, Mapping) else {}

        llm = self.config.get("llm", {})
        sandbox = llm.get("sandbox", {}) if isinstance(llm, Mapping) else {}
        sandbox_map = dict(sandbox) if isinstance(sandbox, Mapping) else {}

        return {
            "enabled": bool(resolved.get("enabled", True)),
            "agentic": bool(resolved.get("agentic", True)),
            "max_steps": int(resolved.get("max_steps", 8)),
            "image_name": str(resolved.get("image_name", DEFAULT_TRAINING_IMAGE)),
            "build_new_image": bool(resolved.get("build_new_image", False)),
            "memory_limit": str(resolved.get("memory_limit", sandbox_map.get("memory_limit", DEFAULT_DOCKER_MEMORY_LIMIT))),
            "cpu_limit": float(resolved.get("cpu_limit", sandbox_map.get("cpu_limit", DEFAULT_DOCKER_CPU_LIMIT))),
            "pids_limit": int(resolved.get("pids_limit", sandbox_map.get("pids_limit", DEFAULT_DOCKER_PIDS_LIMIT))),
            "shm_size": str(resolved.get("shm_size", sandbox_map.get("shm_size", DEFAULT_DOCKER_SHM_SIZE))),
            "host": str(resolved.get("host", sandbox_map.get("host", "127.0.0.1"))),
            "port": int(resolved.get("port", sandbox_map.get("port", 8892))),
        }

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
        test_size = max(1, min(len(shuffled) - 2, int(round(len(shuffled) * self.test_size))))
        train_df = shuffled.iloc[:-test_size].copy()
        val_df = shuffled.iloc[-test_size:].copy()

        train_unique = train_df[target_column].dropna().apply(_normalize_label_value).nunique()
        if train_unique < 2:
            return frame.copy(), frame.iloc[0:0].copy()
        return train_df, val_df

    def _compose_texts(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
        return _compose_texts_from_frame(frame, feature_columns)

    def _target_labels(self, frame: pd.DataFrame, target_column: str) -> list[str]:
        return [_normalize_label_value(value) for value in frame[target_column].tolist()]

    def _uncertainty_scores(self, probabilities: np.ndarray, strategy: str) -> np.ndarray:
        if len(probabilities) == 0:
            return np.asarray([], dtype=np.float32)
        if strategy == "random":
            rng = np.random.default_rng(self.random_seed)
            return rng.random(len(probabilities))
        if strategy == "margin":
            sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
            return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        epsilon = 1e-12
        entropy = -(probabilities * np.log(probabilities + epsilon)).sum(axis=1)
        return entropy

    def _record_log(self, message: str, logs: list[str]) -> None:
        logs.append(message)

    def _merge_logs(self, upstream_logs: Sequence[str], logs: Sequence[str]) -> list[str]:
        return [*upstream_logs, *logs]

    def _accuracy(self, truth: Sequence[str], predicted: Sequence[str]) -> float:
        return _accuracy(truth, predicted)

    def _macro_f1(self, truth: Sequence[str], predicted: Sequence[str]) -> float:
        return _macro_f1(truth, predicted)

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


def _main_training_script() -> int:
    parser = argparse.ArgumentParser(description="Run ActiveLearning training from config")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()
    run_training_job(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_training_script())
