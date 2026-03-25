import argparse
import json
import os
import pickle
import random
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
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_EPOCHS = 20
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_VOCAB = 5000
DEFAULT_MIN_TOKEN_FREQ = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_MAX_STEPS = 8


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


def _default_training_config() -> dict[str, Any]:
    return {
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_vocab": DEFAULT_MAX_VOCAB,
        "min_token_freq": DEFAULT_MIN_TOKEN_FREQ,
    }


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

    task_prompt = str(payload.get("task_prompt") or "").strip()
    train_texts = _compose_texts_from_frame(train_df, feature_columns, task_prompt=task_prompt)
    train_labels = [_normalize_label_value(item) for item in train_df[target_column].tolist()]
    val_texts = _compose_texts_from_frame(val_df, feature_columns, task_prompt=task_prompt)
    val_labels = [_normalize_label_value(item) for item in val_df[target_column].tolist()] if not val_df.empty else []

    model, metrics, checkpoint = train_text_classifier(
        train_texts,
        train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        training_config=_default_training_config(),
        random_seed=int(payload.get("random_seed", 13)),
    )

    model_path = Path(payload["model_path"])
    metrics_path = Path(payload["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


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
        if prefix:
            texts.append(prefix + body)
        else:
            texts.append(body)
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

        learning_curves = self._build_learning_curve_from_training_metrics(metrics)
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
        train_texts = self._compose_texts(train_df, selection.feature_columns, task_prompt=selection.task_prompt)
        train_labels = self._target_labels(train_df, selection.target_column)
        val_texts = self._compose_texts(val_df, selection.feature_columns, task_prompt=selection.task_prompt)
        val_labels = self._target_labels(val_df, selection.target_column) if not val_df.empty else []
        model, metrics, _ = train_text_classifier(
            train_texts,
            train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            training_config=_default_training_config(),
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
        texts = self._compose_texts(working, selection.feature_columns, task_prompt=selection.task_prompt)
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
        predicted = estimator.predict(
            self._compose_texts(evaluation_frame, selection.feature_columns, task_prompt=selection.task_prompt)
        )
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

    def _build_learning_curve_from_training_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> dict[str, list[dict[str, float | int | str]]]:
        strategy_history = self._normalize_curve_history(
            metrics.get("history")
            or metrics.get("learning_curve")
            or metrics.get("curve")
            or metrics.get("strategy_history"),
            strategy=self.query_strategy,
        )
        random_history = self._normalize_curve_history(
            metrics.get("random_history") or metrics.get("baseline_history") or metrics.get("random_curve"),
            strategy="random",
        )
        if strategy_history:
            return {"strategy_history": strategy_history, "random_history": random_history}

        # Fallback: make a single-point curve from top-level training metrics.
        macro_f1 = metrics.get("macro_f1")
        accuracy = metrics.get("accuracy")
        train_rows = metrics.get("train_rows")
        if macro_f1 is None or train_rows is None:
            return {"strategy_history": [], "random_history": random_history}
        single_point = {
            "iteration": 1,
            "n_labeled": int(train_rows),
            "accuracy": float(accuracy if accuracy is not None else 0.0),
            "macro_f1": float(macro_f1),
            "strategy": self.query_strategy,
        }
        return {"strategy_history": [single_point], "random_history": random_history}

    @staticmethod
    def _normalize_curve_history(
        raw_history: Any,
        *,
        strategy: str,
    ) -> list[dict[str, float | int | str]]:
        if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes)):
            return []
        normalized: list[dict[str, float | int | str]] = []
        for index, row in enumerate(raw_history, start=1):
            if not isinstance(row, Mapping):
                continue
            n_labeled = row.get("n_labeled", row.get("labeled_rows", row.get("train_rows")))
            macro_f1 = row.get("macro_f1", row.get("f1", row.get("val_macro_f1", row.get("val_f1"))))
            accuracy = row.get("accuracy", row.get("val_accuracy", 0.0))
            if n_labeled is None or macro_f1 is None:
                continue
            normalized.append(
                {
                    "iteration": int(row.get("iteration", index)),
                    "n_labeled": int(n_labeled),
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                    "strategy": str(row.get("strategy", strategy)),
                }
            )
        return normalized

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
        self._run_training_locally_with_code_agent(artifacts=artifacts)

        if not artifacts.training_metrics_path.exists():
            raise RuntimeError("Generated training script did not produce training metrics.")
        if not artifacts.model_path.exists():
            raise RuntimeError("Generated training script did not produce model checkpoint.")

        # Generated training code may save a torch-native checkpoint that is not
        # compatible with CheckpointClassifier.load(). If so, regenerate outputs
        # with the built-in deterministic training entrypoint.
        try:
            CheckpointClassifier.load(artifacts.model_path)
        except Exception as error:
            config_path = self.output_dir / "training_job_config.json"
            run_training_job(config_path)
            self._record_log(
                "Generated checkpoint format was incompatible; regenerated checkpoint via built-in run_training_job.",
                logs,
            )
            try:
                CheckpointClassifier.load(artifacts.model_path)
            except Exception as second_error:
                raise RuntimeError(
                    "Training produced an incompatible model checkpoint format. "
                    "Expected CheckpointClassifier-compatible payload."
                ) from second_error

        metrics = json.loads(artifacts.training_metrics_path.read_text(encoding="utf-8"))
        self._record_log(
            f"Generated training script executed via local CodeAgent: {artifacts.train_script_path}",
            logs,
        )
        return metrics

    def _run_training_locally_with_code_agent(
        self,
        *,
        artifacts: ActiveLearningArtifacts,
    ) -> None:
        try:
            import httpx
            from smolagents import CodeAgent, OpenAIModel
            from smolagents.agents import RunResult
            from agents.tools import build_search_tools
        except Exception as error:  # pragma: no cover - dependency failure fallback
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
            "Do not use Python `with` context managers in generated code because the local executor may fail on them. "
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
                "fasttext",
                "fasttext.*",
                "json",
                "pathlib",
                "pickle",
                "numpy",
                "numpy.*",
                "pandas",
                "torch",
                "torch.*",
            ],
            max_steps=max_steps,
            verbosity_level=1,
            instructions=instructions,
        ) as agent:
            run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
            if not isinstance(run_result, RunResult):
                raise RuntimeError("Agentic local training returned an unexpected result type.")
            output_payload = self._parse_agent_output(run_result.output)
            if output_payload is not None and output_payload.get("success") is False:
                raise RuntimeError(f"Agentic local training reported failure: {output_payload}")

    def _build_agentic_training_task(
        self,
        *,
        script_path: str,
        config_path: str,
        task_prompt: str,
    ) -> str:
        task_line = task_prompt if task_prompt else "No explicit task prompt provided."
        return (
            "Train an active-learning classifier using the provided config and dataset files.\n"
            f"Config JSON path: {config_path}\n"
            f"Output script path to create/update: {script_path}\n"
            f"Primary user task_prompt (highest priority): {task_line}\n"
            "Requirements:\n"
            "1) Read config JSON and train a torch-based classifier.\n"
            "2) Treat `task_prompt` from config as primary task context and include it in model input construction.\n"
            "3) Write checkpoint to `model_path` and metrics JSON to `metrics_path` from config.\n"
            "4) If first implementation fails, debug using errors and rerun.\n"
            "5) If blocked, use `github_code_search` first and `web_search` second to find reliable training patterns.\n"
            "6) Keep script deterministic with explicit random seeds.\n"
            "7) If using class-based models, prefer explicit `super(CurrentClass, self)` over zero-arg `super()`.\n"
            "8) Do not use any `with` blocks (for example `with torch.no_grad()` or `with torch.inference_mode()`). "
            "Run validation without context managers.\n"
            "9) Return final JSON with keys: success (bool), model_path, metrics_path, notes.\n"
            "Use only Python code execution and finish with final_answer(json.dumps(...))."
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
        test_size = max(1, min(len(shuffled) - 2, int(round(len(shuffled) * self.test_size))))
        train_df = shuffled.iloc[:-test_size].copy()
        val_df = shuffled.iloc[-test_size:].copy()

        train_unique = train_df[target_column].dropna().apply(_normalize_label_value).nunique()
        if train_unique < 2:
            return frame.copy(), frame.iloc[0:0].copy()
        return train_df, val_df

    def _compose_texts(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        task_prompt: str | None = None,
    ) -> list[str]:
        return _compose_texts_from_frame(frame, feature_columns, task_prompt=task_prompt)

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
