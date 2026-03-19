import os
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from ..base import AgentResult, BaseAgent
from models import BaseModelAdapter, OllamaAdapter


DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_TASK = "classification"
SUPPORTED_MODALITIES = {"text", "audio", "image"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_SENTIMENT_KEYWORDS = {
    "positive": {
        "good",
        "great",
        "excellent",
        "love",
        "liked",
        "amazing",
        "happy",
        "best",
        "fantastic",
    },
    "negative": {
        "bad",
        "terrible",
        "awful",
        "hate",
        "poor",
        "worst",
        "boring",
        "slow",
        "broken",
    },
    "neutral": {
        "okay",
        "average",
        "normal",
        "mixed",
        "fine",
        "standard",
        "typical",
    },
}


@dataclass(slots=True)
class AnnotationArtifacts:
    annotated_dataset_path: Path
    spec_path: Path
    quality_path: Path
    labelstudio_path: Path
    review_path: Path


class DataAnnotationAgent(BaseAgent):
    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        output_dir: str | Path = "data/raw",
        modality: str | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_config = self._resolve_annotation_config()
        self.llm_config = self._resolve_llm_config()
        self.process_config = self._resolve_process_config()
        self.modality = self._resolve_modality(modality)
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else self.annotation_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
        self.task = str(self.annotation_config.get("task", DEFAULT_TASK)).strip() or DEFAULT_TASK
        self.label_column = str(self.annotation_config.get("label_column", "label")).strip() or "label"
        self.overwrite_existing_labels = bool(self.annotation_config.get("overwrite_existing_labels", False))
        self._model_adapter: BaseModelAdapter | None = None

    def run(
        self,
        dataframe: pd.DataFrame,
        task: str | None = None,
        annotation_prompt: str | None = None,
    ) -> pd.DataFrame:
        result = self.execute(
            {
                "dataframe": dataframe,
                "task": task or self.task,
                "annotation_prompt": annotation_prompt,
            }
        )
        if result.dataframe is None:
            raise RuntimeError("DataAnnotationAgent did not produce a dataframe.")
        return result.dataframe

    def execute(self, payload: Any | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting DataAnnotationAgent execution.", logs)

        upstream = payload if isinstance(payload, AgentResult) else None
        source_frame = self._resolve_dataframe(payload)
        task = self._resolve_task(payload)
        annotation_prompt = self._resolve_annotation_prompt(payload)
        labeled = self.auto_label(source_frame, self.modality, prompt=annotation_prompt)
        quality_metrics = self.check_quality(labeled)
        spec_text = self.generate_spec(labeled, task=task)
        labelstudio_payload = self.export_to_labelstudio(labeled)
        review_payload = self.flag_for_review(labeled, threshold=self.confidence_threshold)
        artifacts = self._write_artifacts(
            labeled=labeled,
            spec_text=spec_text,
            quality_metrics=quality_metrics,
            labelstudio_payload=labelstudio_payload,
            review_payload=review_payload,
        )

        self._record_log(f"Wrote annotated dataset to {artifacts.annotated_dataset_path}.", logs)
        self._record_log(f"Wrote annotation spec to {artifacts.spec_path}.", logs)
        self._record_log(f"Wrote annotation quality report to {artifacts.quality_path}.", logs)
        self._record_log(f"Wrote Label Studio import file to {artifacts.labelstudio_path}.", logs)
        self._record_log(f"Wrote review queue to {artifacts.review_path}.", logs)
        self._record_log("DataAnnotationAgent execution finished successfully.", logs)

        schema = {column: str(dtype) for column, dtype in labeled.dtypes.items()}
        upstream_artifacts = dict(upstream.artifacts) if upstream is not None else {}
        upstream_metadata = deepcopy(upstream.metadata) if upstream is not None else {}
        upstream_logs = list(upstream.logs) if upstream is not None else []
        upstream_metrics = dict(upstream.metrics) if upstream is not None else {}

        merged_artifacts = {
            **upstream_artifacts,
            "annotated_dataset": str(artifacts.annotated_dataset_path),
            "annotation_spec": str(artifacts.spec_path),
            "annotation_quality": str(artifacts.quality_path),
            "labelstudio_import": str(artifacts.labelstudio_path),
            "low_confidence_review": str(artifacts.review_path),
        }
        merged_metadata = {
            **upstream_metadata,
            "annotation": {
                "task": task,
                "modality": self.modality,
                "confidence_threshold": self.confidence_threshold,
                "prompt": annotation_prompt,
                "quality": quality_metrics,
                "low_confidence_count": len(review_payload),
            },
        }
        merged_metrics = {
            **upstream_metrics,
            "row_count": int(len(labeled)),
            "annotation_confidence_mean": quality_metrics.get("confidence_mean"),
            "annotation_kappa": quality_metrics.get("kappa"),
            "annotation_low_confidence_count": len(review_payload),
        }
        merged_logs = self._merge_logs(upstream_logs, logs)

        return AgentResult(
            dataframe=labeled,
            dataframe_path=artifacts.annotated_dataset_path,
            dataframe_schema=schema,
            metrics=merged_metrics,
            artifacts=merged_artifacts,
            logs=merged_logs,
            metadata=merged_metadata,
        )

    def auto_label(
        self,
        df: pd.DataFrame,
        modality: str | None = None,
        prompt: str | None = None,
    ) -> pd.DataFrame:
        effective_modality = self._resolve_modality(modality)
        annotation_prompt = self._normalize_optional_text(prompt)
        if annotation_prompt:
            return self._auto_label_with_prompt(df, effective_modality, annotation_prompt)
        working = df.copy()
        signal_column = self._signal_column(effective_modality)
        if signal_column not in working.columns:
            working[signal_column] = None
        if self.label_column not in working.columns:
            working[self.label_column] = None

        original_labels = pd.Series(
            [self._normalize_optional_text(value) for value in working[self.label_column].tolist()],
            index=working.index,
            dtype=object,
        )
        if not self._heuristic_labeling_enabled(original_labels):
            return self._pass_through_annotation(working, effective_modality)
        prototypes = self._build_label_prototypes(working, signal_column, original_labels)
        fallback_label = self._fallback_label(prototypes, original_labels)

        auto_labels: list[str | None] = []
        confidences: list[float] = []
        for _, row in working.iterrows():
            label, confidence = self._predict_row_label(
                row=row,
                signal_column=signal_column,
                prototypes=prototypes,
                fallback_label=fallback_label,
            )
            auto_labels.append(label)
            confidences.append(confidence)

        working["annotation_reference_label"] = original_labels
        working["annotation_auto_label"] = auto_labels
        working["annotation_confidence"] = confidences
        working["annotation_label_origin"] = [
            "auto" if reference is None else "existing" for reference in original_labels
        ]
        final_labels = [
            auto_label
            if self.overwrite_existing_labels or reference is None
            else reference
            for reference, auto_label in zip(original_labels, auto_labels, strict=False)
        ]
        working[self.label_column] = final_labels
        working["annotation_needs_review"] = [
            bool(confidence < self.confidence_threshold or (reference and auto_label and reference != auto_label))
            for reference, auto_label, confidence in zip(
                original_labels,
                auto_labels,
                confidences,
                strict=False,
            )
        ]
        return working

    def _pass_through_annotation(self, df: pd.DataFrame, modality: str) -> pd.DataFrame:
        working = df.copy()
        signal_column = self._signal_column(modality)
        if signal_column not in working.columns:
            working[signal_column] = None
        if self.label_column not in working.columns:
            working[self.label_column] = None
        original_labels = pd.Series(
            [self._normalize_optional_text(value) for value in working[self.label_column].tolist()],
            index=working.index,
            dtype=object,
        )
        working["annotation_reference_label"] = original_labels
        working["annotation_auto_label"] = original_labels
        working["annotation_confidence"] = [1.0 if label is not None else 0.0 for label in original_labels]
        working["annotation_label_origin"] = [
            "existing" if label is not None else "unresolved" for label in original_labels
        ]
        working[self.label_column] = original_labels
        working["annotation_needs_review"] = [label is None for label in original_labels]
        return working

    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        class_defs = self._resolve_class_definitions(df)
        label_examples = self._label_examples(df, class_defs)
        ambiguous_examples = self._ambiguous_examples(df)
        objective = (
            str(self.annotation_config.get("objective", "")).strip()
            or f"Assign `{self.label_column}` labels for the `{task}` task."
        )
        lines = [
            "# Annotation Specification\n",
            "\n",
            "## Task\n",
            "\n",
            f"- Task: `{task}`\n",
            f"- Objective: {objective}\n",
            f"- Modality: `{self.modality}`\n",
            f"- Target column: `{self.label_column}`\n",
            f"- Human review threshold: `{self.confidence_threshold:.2f}`\n",
            "\n",
            "## Classes\n",
            "\n",
        ]
        if not class_defs:
            lines.append("- No classes were inferred. Samples without usable predictions should be reviewed manually.\n")
        for class_def in class_defs:
            name = class_def["name"]
            description = class_def.get("description") or "No explicit description configured."
            keywords = ", ".join(class_def.get("keywords", [])) or "none"
            lines.extend(
                [
                    f"### {name}\n",
                    "\n",
                    f"- Definition: {description}\n",
                    f"- Auto-label keywords: {keywords}\n",
                    "- Examples:\n",
                ]
            )
            for example in label_examples.get(name, []):
                lines.append(f"  - {example}\n")
            lines.append("\n")

        lines.extend(
            [
                "## Edge Cases\n",
                "\n",
                "- Empty or missing modality fields should go to manual review.\n",
                "- Rows where the existing label disagrees with the auto-label should be reviewed.\n",
                "- Low-confidence predictions should be exported for human validation.\n",
                "\n",
                "## Ambiguous Examples\n",
                "\n",
            ]
        )
        if ambiguous_examples:
            for example in ambiguous_examples:
                lines.append(f"- {example}\n")
        else:
            lines.append("- When the content is ambiguous or mixed, send it to manual review instead of forcing a label.\n")
        return "".join(lines)

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        confidence_series = self._confidence_series(df_labeled)
        valid_confidence = confidence_series.dropna()
        label_series = df_labeled.get(self.label_column)
        label_dist = (
            {}
            if label_series is None
            else {
                str(label): int(count)
                for label, count in label_series.dropna().astype(str).value_counts().sort_index().items()
            }
        )
        reference = df_labeled.get("annotation_reference_label")
        auto = df_labeled.get("annotation_auto_label")
        kappa = None
        agreement = None
        if reference is not None and auto is not None:
            comparable = pd.DataFrame({"reference": reference, "auto": auto}).dropna()
            comparable = comparable[
                comparable["reference"].astype(str).str.strip().ne("")
                & comparable["auto"].astype(str).str.strip().ne("")
            ]
            if len(comparable) >= 2:
                kappa = self._cohen_kappa(
                    comparable["reference"].astype(str).tolist(),
                    comparable["auto"].astype(str).tolist(),
                )
                agreement = float(
                    (comparable["reference"].astype(str) == comparable["auto"].astype(str)).mean()
                )
        review_series = df_labeled.get("annotation_needs_review")
        if review_series is None:
            review_series = pd.Series(False, index=df_labeled.index, dtype=bool)
        review_count = int(review_series.fillna(False).sum())
        return {
            "kappa": kappa,
            "agreement_pct": None if agreement is None else float(agreement * 100.0),
            "agreement_rate": agreement,
            "label_dist": label_dist,
            "confidence_mean": None if valid_confidence.empty else float(valid_confidence.mean()),
            "review_count": review_count,
            "review_rate": 0.0 if len(df_labeled) == 0 else float(review_count / len(df_labeled)),
        }

    def export_to_labelstudio(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        signal_column = self._signal_column(self.modality)
        tasks: list[dict[str, Any]] = []
        for index, row in df.reset_index(drop=True).iterrows():
            signal_value = row.get(signal_column)
            if self._is_empty(signal_value):
                continue
            task: dict[str, Any] = {
                "id": int(index + 1),
                "data": {signal_column: signal_value},
                "meta": {
                    "source": row.get("source"),
                    "confidence": self._safe_float(row.get("annotation_confidence")),
                    "needs_review": bool(row.get("annotation_needs_review", False)),
                },
            }
            annotation = self._labelstudio_annotation(row, signal_column)
            if annotation is not None:
                task["annotations"] = [annotation]
            tasks.append(task)
        return tasks

    def flag_for_review(
        self,
        df_labeled: pd.DataFrame,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        effective_threshold = float(threshold if threshold is not None else self.confidence_threshold)
        review_mask = self._confidence_series(df_labeled).fillna(0.0) < effective_threshold
        if "annotation_reference_label" in df_labeled.columns and "annotation_auto_label" in df_labeled.columns:
            disagreement_mask = (
                df_labeled["annotation_reference_label"].notna()
                & df_labeled["annotation_auto_label"].notna()
                & df_labeled["annotation_reference_label"].astype(str).ne(
                    df_labeled["annotation_auto_label"].astype(str)
                )
            )
            review_mask = review_mask | disagreement_mask

        review_rows = df_labeled[review_mask].copy()

        tasks = self.export_to_labelstudio(review_rows)
        for task, (_, row) in zip(tasks, review_rows.iterrows(), strict=False):
            task["meta"]["review_reason"] = self._review_reason(row, effective_threshold)
        return tasks

    def process(self, df_path: str | Path, prompt: str) -> Path:
        source_path = Path(df_path)
        frame = self._read_dataframe(source_path)
        processed, _ = self._process_frame(frame, prompt=prompt)
        if processed.equals(frame):
            output_path = source_path.with_name(f"{source_path.stem}_processed.csv")
            frame.to_csv(output_path, index=False)
            return output_path

        output_path = source_path.with_name(f"{source_path.stem}_processed.csv")
        processed.to_csv(output_path, index=False)
        return output_path

    def _auto_label_with_prompt(
        self,
        df: pd.DataFrame,
        modality: str,
        prompt: str,
    ) -> pd.DataFrame:
        working = df.copy()
        signal_column = self._signal_column(modality)
        if signal_column not in working.columns:
            working[signal_column] = None
        if self.label_column not in working.columns:
            working[self.label_column] = None

        original_labels = pd.Series(
            [self._normalize_optional_text(value) for value in working[self.label_column].tolist()],
            index=working.index,
            dtype=object,
        )
        candidate_mask = pd.Series(True, index=working.index, dtype=bool)
        if not self.overwrite_existing_labels:
            candidate_mask = original_labels.isna()

        processed = working.copy()
        processed_mask = pd.Series(False, index=working.index, dtype=bool)
        filtered = working.loc[candidate_mask].copy()
        if not filtered.empty:
            row_prompt = self._build_annotation_row_prompt(prompt)
            transformed, successful_indices = self._transform_rows(
                filtered,
                row_prompt,
                json_schema=self._annotation_result_schema(prompt),
            )
            allowed_columns = set(working.columns) | self._allowed_prompt_output_columns(prompt)
            transformed = transformed[[column for column in transformed.columns if column in allowed_columns]]
            for column in transformed.columns:
                if column not in processed.columns:
                    processed[column] = None
                elif processed[column].dtype != object:
                    processed[column] = processed[column].astype(object)
            processed.loc[transformed.index, transformed.columns] = transformed
            processed_mask.loc[successful_indices] = True
        if self.label_column not in processed.columns:
            processed[self.label_column] = original_labels

        prompt_auto_column = processed["annotation_auto_label"] if "annotation_auto_label" in processed.columns else None
        prompt_confidence_column = (
            pd.to_numeric(processed["annotation_confidence"], errors="coerce")
            if "annotation_confidence" in processed.columns
            else pd.Series(index=processed.index, dtype=float)
        )

        auto_labels: list[str | None] = []
        confidences: list[float] = []
        label_origins: list[str] = []
        final_labels: list[str | None] = []
        needs_review: list[bool] = []

        for index in processed.index:
            reference = original_labels.loc[index]
            current_label = self._normalize_optional_text(processed.at[index, self.label_column])
            prompt_auto = None if prompt_auto_column is None else self._normalize_optional_text(prompt_auto_column.loc[index])
            has_complete_problem = self._safe_bool(processed.at[index, "has_complete_problem"]) if "has_complete_problem" in processed.columns else None
            if has_complete_problem is False:
                current_label = None
                prompt_auto = None
            auto_label = prompt_auto or current_label or reference
            confidence = self._safe_float(prompt_confidence_column.loc[index])
            if confidence is None:
                if processed_mask.loc[index]:
                    confidence = 1.0 if auto_label is not None or has_complete_problem is False else 0.0
                else:
                    confidence = 1.0 if reference is not None else 0.0
            if processed_mask.loc[index]:
                label_origin = "prompt"
            elif reference is not None:
                label_origin = "existing"
            else:
                label_origin = "unresolved"
            final_label = current_label if current_label is not None else reference
            incomplete_without_label = final_label is None and has_complete_problem is False

            auto_labels.append(auto_label)
            confidences.append(float(confidence))
            label_origins.append(label_origin)
            final_labels.append(final_label)
            needs_review.append(
                bool(
                    (final_label is None and not incomplete_without_label)
                    or confidence < self.confidence_threshold
                    or (reference is not None and auto_label is not None and reference != auto_label)
                )
            )

        processed["annotation_reference_label"] = original_labels
        processed["annotation_auto_label"] = auto_labels
        processed["annotation_confidence"] = confidences
        processed["annotation_label_origin"] = label_origins
        processed[self.label_column] = final_labels
        processed["annotation_needs_review"] = needs_review
        return processed

    def _build_annotation_row_prompt(self, prompt: str) -> str:
        lines = [
            "You are annotating dataset rows represented as JSON objects.",
            f"User task: {prompt.strip()}",
            f"Target label column: `{self.label_column}`.",
            "Return only a JSON object for the same row.",
            "Preserve all original keys unless you are explicitly adding a new field required by the task.",
            "Do not include explanations, markdown, or chain-of-thought.",
            "If you cannot determine a valid label because the problem is incomplete or missing required context, leave the label empty instead of guessing.",
            "If a row refers to a missing passage, dialogue, audio, image, table, or 'the text' without including that material in the row itself, treat the problem as incomplete.",
            f"If `{self.label_column}` is empty, do not place explanations or reasoning in `{self.label_column}`.",
        ]
        if "has_complete_problem" in prompt:
            lines.append(
                "When the task asks for `has_complete_problem`, set it to false for incomplete/unsolved rows and true otherwise."
            )
            lines.append(
                f"If `has_complete_problem` is true, `{self.label_column}` must contain the concrete answer. "
                f"If you cannot provide the answer, set `{self.label_column}` to null and `has_complete_problem` to false."
            )
        return "\n".join(lines)

    def _annotation_result_schema(self, prompt: str) -> dict[str, Any]:
        properties: dict[str, Any] = {
            self.label_column: {
                "type": ["string", "number", "boolean", "null"],
            }
        }
        required = [self.label_column]
        if "has_complete_problem" in prompt:
            properties["has_complete_problem"] = {"type": ["boolean", "null"]}
            required.append("has_complete_problem")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }

    def _batch_transform_schema(self, row_schema: Mapping[str, Any] | None) -> dict[str, Any]:
        item_schema = dict(row_schema or {"type": "object"})
        properties = dict(item_schema.get("properties", {}))
        properties["_process_row_index"] = {"type": "string"}
        required = list(item_schema.get("required", []))
        if "_process_row_index" not in required:
            required.append("_process_row_index")
        return {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["rows"],
            "additionalProperties": False,
        }

    def _allowed_prompt_output_columns(self, prompt: str) -> set[str]:
        allowed = {
            self.label_column,
            "annotation_confidence",
        }
        allowed.update(self._prompt_declared_fields(prompt))
        return allowed

    def _prompt_declared_fields(self, prompt: str) -> set[str]:
        fields: set[str] = set()
        for match in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', prompt):
            fields.add(match)
        for match in re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", prompt):
            fields.add(match)
        return fields

    def _process_frame(
        self,
        frame: pd.DataFrame,
        *,
        prompt: str,
        candidate_mask: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        working = frame.copy()
        effective_mask = (
            pd.Series(True, index=working.index, dtype=bool)
            if candidate_mask is None
            else candidate_mask.reindex(working.index, fill_value=False).astype(bool)
        )
        eligible = working.loc[effective_mask].copy()
        processed_mask = pd.Series(False, index=working.index, dtype=bool)
        if eligible.empty:
            return working, processed_mask

        selected = self._select_rows_for_process(eligible, prompt)
        filtered = eligible.iloc[selected["row_positions"]].copy()
        if filtered.empty:
            return working, processed_mask

        row_prompt = self._build_row_prompt(filtered, prompt)
        transformed, successful_indices = self._transform_rows(filtered, row_prompt)
        for column in transformed.columns:
            if column not in working.columns:
                working[column] = None
            elif working[column].dtype != object:
                working[column] = working[column].astype(object)
        working.loc[transformed.index, transformed.columns] = transformed
        processed_mask.loc[successful_indices] = True
        return working, processed_mask

    def _select_rows_for_process(self, frame: pd.DataFrame, prompt: str) -> dict[str, Any]:
        sample_size = int(self.process_config.get("sample_size", 8))
        sample_rows = []
        for row_position, (_, row) in enumerate(frame.head(sample_size).iterrows()):
            payload = self._json_ready(row.to_dict())
            payload["_row_position"] = row_position
            sample_rows.append(payload)

        selection = self._chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You select which DataFrame rows should be transformed. "
                        "Prefer a reusable pandas-style boolean expression in `filter_condition` "
                        "when possible. Use explicit `row_indices` only when the match is exceptional."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt": prompt,
                            "schema": list(frame.columns),
                            "sample_rows": sample_rows,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_schema={
                "type": "object",
                "properties": {
                    "filter_condition": {"type": ["string", "null"]},
                    "row_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["filter_condition", "row_indices"],
                "additionalProperties": False,
            },
        )
        return {
            "filter_condition": selection.get("filter_condition"),
            "row_positions": self._apply_process_selection(frame, selection),
        }

    def _build_row_prompt(self, filtered: pd.DataFrame, prompt: str) -> str:
        response = self._chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user request into a deterministic instruction for a single JSON row. "
                        "The instruction must preserve unchanged keys unless the user explicitly asks otherwise."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_prompt": prompt,
                            "sample_filtered_row": self._json_ready(filtered.iloc[0].to_dict()),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_schema={
                "type": "object",
                "properties": {"row_prompt": {"type": "string"}},
                "required": ["row_prompt"],
                "additionalProperties": False,
            },
        )
        row_prompt = str(response.get("row_prompt", "")).strip()
        if not row_prompt:
            raise ValueError("Row transformation prompt generation returned an empty prompt.")
        return row_prompt

    def _transform_rows(
        self,
        filtered: pd.DataFrame,
        row_prompt: str,
        json_schema: Mapping[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, list[Any]]:
        if filtered.empty:
            return filtered.copy(), []
        parallel_workers = self._effective_parallel_workers()
        batch_size = self._effective_batch_size()
        transformed_rows: dict[Any, dict[str, Any]] = {}
        successful_indices: list[Any] = []
        batches = [
            filtered.iloc[start : start + batch_size].copy()
            for start in range(0, len(filtered), batch_size)
        ]
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_batch = {
                executor.submit(self._transform_row_batch, batch, row_prompt, json_schema): tuple(batch.index.tolist())
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch_rows, batch_successes = future.result()
                transformed_rows.update(batch_rows)
                successful_indices.extend(batch_successes)
        transformed = pd.DataFrame.from_dict(transformed_rows, orient="index")
        transformed = transformed.reindex(filtered.index)
        if "_process_row_index" in transformed.columns:
            transformed = transformed.drop(columns=["_process_row_index"])
        return transformed, successful_indices

    def _transform_row_batch(
        self,
        batch: pd.DataFrame,
        row_prompt: str,
        json_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[Any, dict[str, Any]], list[Any]]:
        original_rows = {index: dict(row.to_dict()) for index, row in batch.iterrows()}
        if len(batch) == 1:
            index = next(iter(original_rows))
            transformed_row, success = self._transform_single_row(index, original_rows[index], row_prompt, json_schema)
            return {index: transformed_row}, [index] if success else []

        timeout_per_row = int(self.process_config.get("timeout_per_row", self.llm_config.get("timeout", 60)))
        batch_timeout = max(timeout_per_row, timeout_per_row * len(batch))
        row_id_map = {str(position): index for position, index in enumerate(batch.index.tolist())}
        payload_rows = []
        for row_id, index in row_id_map.items():
            row_payload = self._json_ready(original_rows[index])
            row_payload["_process_row_index"] = row_id
            payload_rows.append(row_payload)
        try:
            response = self._chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{row_prompt}\n"
                            "Return only a JSON object with a `rows` array. Each item must include "
                            "`_process_row_index` copied from the input row. Preserve all original keys unless "
                            "you are explicitly adding a new field. Do not change `id` if it exists."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({"rows": payload_rows}, ensure_ascii=False),
                    },
                ],
                json_schema=self._batch_transform_schema(json_schema),
                timeout=batch_timeout,
            )
            transformed_rows = dict(original_rows)
            successful_indices: list[Any] = []
            for transformed_row in response.get("rows", []):
                if not isinstance(transformed_row, Mapping):
                    continue
                row_id = str(transformed_row.get("_process_row_index", "")).strip()
                if row_id not in row_id_map:
                    continue
                index = row_id_map[row_id]
                normalized_row = dict(transformed_row)
                normalized_row.pop("_process_row_index", None)
                transformed_rows[index] = self._normalize_transformed_row(original_rows[index], normalized_row)
                successful_indices.append(index)
            return transformed_rows, successful_indices
        except Exception:
            transformed_rows = dict(original_rows)
            successful_indices: list[Any] = []
            for index, original_row in original_rows.items():
                transformed_row, success = self._transform_single_row(index, original_row, row_prompt, json_schema)
                transformed_rows[index] = transformed_row
                if success:
                    successful_indices.append(index)
            return transformed_rows, successful_indices

    def _transform_single_row(
        self,
        index: Any,
        row_payload: Mapping[str, Any],
        row_prompt: str,
        json_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        max_retries = max(1, int(self.process_config.get("max_retries", 3)))
        timeout_per_row = int(self.process_config.get("timeout_per_row", self.llm_config.get("timeout", 60)))
        original_row = dict(row_payload)
        json_ready_row = self._json_ready(original_row)
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                response = self._chat_json(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{row_prompt}\n"
                                "Return only a JSON object. Preserve all original keys unless you are explicitly "
                                "adding a new field. Do not change `id` if it exists."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(json_ready_row, ensure_ascii=False),
                        },
                    ],
                    json_schema=json_schema or {"type": "object"},
                    timeout=timeout_per_row,
                )
                return self._normalize_transformed_row(original_row, response), True
            except Exception as error:
                last_error = error
        if last_error is not None:
            self._record_log(
                f"Process transform retries exhausted for row {index}: {type(last_error).__name__}: {last_error}"
            )
        return original_row, False

    def _apply_process_selection(self, frame: pd.DataFrame, selection: Mapping[str, Any]) -> list[int]:
        raw_positions = selection.get("row_indices") or []
        if isinstance(raw_positions, Sequence) and not isinstance(raw_positions, (str, bytes)):
            valid_positions = []
            for value in raw_positions:
                try:
                    position = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= position < len(frame):
                    valid_positions.append(position)
            if valid_positions:
                return sorted(dict.fromkeys(valid_positions))

        filter_condition = str(selection.get("filter_condition") or "").strip()
        if not filter_condition:
            return []
        try:
            mask = frame.eval(filter_condition, engine="python")
            if isinstance(mask, pd.Series):
                return [int(position) for position, include in enumerate(mask.fillna(False).tolist()) if bool(include)]
        except Exception:
            pass
        try:
            queried = frame.query(filter_condition, engine="python")
            return [int(frame.index.get_loc(index)) for index in queried.index]
        except Exception as error:
            raise ValueError(f"Unable to apply LLM filter_condition {filter_condition!r}: {error}") from error

    def _normalize_transformed_row(
        self,
        original_row: Mapping[str, Any],
        transformed_row: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(transformed_row, Mapping):
            raise ValueError("Row transformation did not return a JSON object.")
        normalized = dict(original_row)
        normalized.update(dict(transformed_row))
        if "id" in original_row and normalized.get("id") != original_row.get("id"):
            raise ValueError("Row transformation changed the immutable `id` field.")
        for key in original_row:
            if key not in normalized:
                raise ValueError(f"Row transformation dropped required key {key!r}.")
        return normalized

    def _json_ready(self, value: Any) -> Any:
        if self._is_empty(value):
            return None
        if isinstance(value, Mapping):
            return {str(key): self._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_ready(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return self._json_ready(value.item())
            except Exception:
                return str(value)
        return value

    def _resolve_dataframe(self, payload: Any | None) -> pd.DataFrame:
        if isinstance(payload, AgentResult):
            if payload.dataframe is not None:
                return payload.dataframe.copy()
            if payload.dataframe_path is not None:
                return self._read_dataframe(Path(payload.dataframe_path))
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, Mapping):
            if isinstance(payload.get("dataframe"), pd.DataFrame):
                return payload["dataframe"].copy()
            if payload.get("dataframe_path"):
                return self._read_dataframe(Path(payload["dataframe_path"]))
        input_path = self.annotation_config.get("input_path")
        if input_path:
            return self._read_dataframe(Path(input_path))
        cleaned_path = self.output_dir / "cleaned_dataset.jsonl"
        if cleaned_path.exists():
            return self._read_dataframe(cleaned_path)
        unified_path = self.output_dir / "unified_dataset.jsonl"
        if unified_path.exists():
            return self._read_dataframe(unified_path)
        annotated_path = self.output_dir / "annotated_dataset.jsonl"
        if annotated_path.exists():
            return self._read_dataframe(annotated_path)
        raise ValueError("DataAnnotationAgent requires a dataframe payload or annotation.input_path.")

    def _resolve_task(self, payload: Any | None) -> str:
        if isinstance(payload, Mapping) and isinstance(payload.get("task"), str) and payload.get("task", "").strip():
            return str(payload["task"]).strip()
        configured_task = str(self.annotation_config.get("task", "")).strip()
        if configured_task:
            return configured_task
        project_name = str(self.config.get("project", {}).get("name", "")).strip()
        if project_name:
            return project_name
        return DEFAULT_TASK

    def _resolve_annotation_prompt(self, payload: Any | None) -> str | None:
        if isinstance(payload, Mapping):
            prompt = self._normalize_optional_text(payload.get("annotation_prompt"))
            if prompt:
                return prompt
        for key in ("annotation_prompt", "prompt", "instructions"):
            prompt = self._normalize_optional_text(self.annotation_config.get(key))
            if prompt:
                return prompt
        return None

    def _resolve_annotation_config(self) -> dict[str, Any]:
        agents_config = self.config.get("agents", {})
        if isinstance(agents_config, Mapping) and isinstance(agents_config.get("annotation"), Mapping):
            return dict(agents_config["annotation"])
        annotation_config = self.config.get("annotation", {})
        if isinstance(annotation_config, Mapping):
            return dict(annotation_config)
        return {}

    def _resolve_llm_config(self) -> dict[str, Any]:
        llm_config = self.config.get("llm", {})
        if isinstance(llm_config, Mapping):
            return dict(llm_config)
        return {}

    def _resolve_process_config(self) -> dict[str, Any]:
        configured = self.annotation_config.get("process_config", self.annotation_config.get("process", {}))
        if not isinstance(configured, Mapping):
            configured = {}
        merged = {
            "parallel_workers": 200,
            "remote_parallel_workers": 2,
            "batch_size": configured.get("batch_size"),
            "remote_batch_size": 4,
            "max_tokens": 8192,
            "timeout_per_row": 10,
            "max_retries": 3,
            "sample_size": 8,
            "model": configured.get("model")
            or self.annotation_config.get("model")
            or self.llm_config.get("model")
            or "gpt-4o-mini",
            "base_url": configured.get("base_url")
            or self.annotation_config.get("base_url")
            or configured.get("api_base")
            or self.annotation_config.get("api_base")
            or self.llm_config.get("base_url")
            or self.llm_config.get("api_base")
            or "http://localhost:11434/v1/chat/completions",
            "api_key": configured.get("api_key")
            or self.annotation_config.get("api_key")
            or self.llm_config.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or "ollama",
            "headers": configured.get("headers")
            or self.annotation_config.get("headers")
            or self.llm_config.get("headers")
            or {},
            "timeout": configured.get("timeout")
            or self.annotation_config.get("timeout")
            or self.llm_config.get("timeout", 60),
        }
        merged.update(dict(configured))
        return merged

    def _resolve_modality(self, modality: str | None) -> str:
        candidate = modality or self.annotation_config.get("modality") or self.config.get("project", {}).get("modality")
        normalized = str(candidate or "text").strip().lower()
        if normalized not in SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported annotation modality '{normalized}'.")
        return normalized

    def _signal_column(self, modality: str) -> str:
        column_name = self.annotation_config.get(f"{modality}_column")
        if isinstance(column_name, str) and column_name.strip():
            return column_name.strip()
        return modality

    def _build_label_prototypes(
        self,
        frame: pd.DataFrame,
        signal_column: str,
        labels: pd.Series,
    ) -> dict[str, Counter[str]]:
        prototypes: dict[str, Counter[str]] = {}
        for class_def in self._resolve_class_definitions(frame):
            seed = Counter(self._tokenize(class_def.get("description", "")))
            seed.update(self._tokenize(" ".join(class_def.get("keywords", []))))
            seed.update(self._tokenize(" ".join(class_def.get("examples", []))))
            if class_def["name"].lower() in DEFAULT_SENTIMENT_KEYWORDS:
                seed.update(DEFAULT_SENTIMENT_KEYWORDS[class_def["name"].lower()])
            prototypes[class_def["name"]] = seed

        if signal_column in frame.columns:
            for label in labels.dropna().astype(str).tolist():
                if label not in prototypes:
                    prototypes[label] = Counter()
            for idx, label in labels.items():
                normalized_label = self._normalize_optional_text(label)
                if normalized_label is None:
                    continue
                row = frame.loc[idx]
                prototypes[normalized_label].update(self._row_tokens(row.get(signal_column), row.get("metadata")))
        return prototypes

    def _resolve_class_definitions(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        configured = self.annotation_config.get("classes") or self.annotation_config.get("labels") or []
        definitions: list[dict[str, Any]] = []
        seen: set[str] = set()
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            for item in configured:
                class_def = self._normalize_class_definition(item)
                if class_def is None or class_def["name"] in seen:
                    continue
                definitions.append(class_def)
                seen.add(class_def["name"])

        if self.label_column in frame.columns:
            for label in frame[self.label_column].dropna().astype(str):
                normalized = label.strip()
                if not normalized or normalized in seen:
                    continue
                definitions.append(
                    {
                        "name": normalized,
                        "description": "",
                        "keywords": [],
                        "examples": [],
                    }
                )
                seen.add(normalized)
        return definitions

    def _normalize_class_definition(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                return None
            return {"name": name, "description": "", "keywords": [], "examples": []}
        if not isinstance(item, Mapping):
            return None
        name = str(item.get("name") or item.get("label") or "").strip()
        if not name:
            return None
        keywords = [str(value).strip() for value in item.get("keywords", []) if str(value).strip()]
        examples = [str(value).strip() for value in item.get("examples", []) if str(value).strip()]
        description = str(item.get("description", "")).strip()
        return {
            "name": name,
            "description": description,
            "keywords": keywords,
            "examples": examples,
        }

    def _fallback_label(self, prototypes: Mapping[str, Counter[str]], labels: pd.Series) -> str | None:
        if not labels.dropna().empty:
            counts = labels.dropna().astype(str).value_counts()
            if not counts.empty:
                return str(counts.index[0])
        if prototypes:
            return next(iter(prototypes))
        return None

    def _heuristic_labeling_enabled(self, labels: pd.Series) -> bool:
        observed = [str(label).strip() for label in labels.dropna().tolist() if str(label).strip()]
        if not observed:
            return True
        unique_count = len(set(observed))
        observed_count = len(observed)
        numeric_count = sum(1 for label in observed if self._is_numeric_like(label))
        numeric_ratio = numeric_count / observed_count
        unique_ratio = unique_count / observed_count
        if numeric_ratio >= 0.8 and unique_ratio >= 0.5:
            return False
        if unique_count >= 20 and unique_ratio >= 0.4:
            return False
        return True

    def _is_numeric_like(self, value: str) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def _predict_row_label(
        self,
        *,
        row: pd.Series,
        signal_column: str,
        prototypes: Mapping[str, Counter[str]],
        fallback_label: str | None,
    ) -> tuple[str | None, float]:
        tokens = self._row_tokens(row.get(signal_column), row.get("metadata"))
        if not tokens:
            return fallback_label, 0.0
        scores: dict[str, float] = {}
        token_counts = Counter(tokens)
        for label, prototype in prototypes.items():
            if not prototype:
                continue
            overlap = 0.0
            for token, count in token_counts.items():
                overlap += count * float(prototype.get(token, 0))
            if label.lower() in DEFAULT_SENTIMENT_KEYWORDS:
                overlap += float(sum(1 for token in tokens if token in DEFAULT_SENTIMENT_KEYWORDS[label.lower()]))
            if overlap > 0:
                scores[label] = overlap

        if not scores:
            return fallback_label, 0.0 if fallback_label is None else 0.34

        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        total_score = sum(scores.values())
        confidence = 1.0 if total_score <= 0 else min(1.0, float(best_score / total_score))
        return best_label, confidence

    def _label_examples(
        self,
        frame: pd.DataFrame,
        class_defs: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[str]]:
        signal_column = self._signal_column(self.modality)
        examples: dict[str, list[str]] = {
            str(class_def["name"]): [str(example).strip() for example in class_def.get("examples", []) if str(example).strip()]
            for class_def in class_defs
        }
        if signal_column not in frame.columns or self.label_column not in frame.columns:
            return examples
        for _, row in frame[[signal_column, self.label_column]].dropna().iterrows():
            label = str(row[self.label_column]).strip()
            signal = str(row[signal_column]).strip()
            if not label or not signal:
                continue
            examples.setdefault(label, [])
            if signal[:160] not in examples[label]:
                examples[label].append(signal[:160])
        for label, entries in examples.items():
            while len(entries) < 3:
                if entries:
                    entries.append(entries[len(entries) % len(entries)])
                else:
                    entries.append(f"No confirmed example available yet for `{label}`.")
        return examples

    def _ambiguous_examples(self, frame: pd.DataFrame) -> list[str]:
        signal_column = self._signal_column(self.modality)
        examples: list[str] = []
        if signal_column not in frame.columns:
            return examples
        review_mask = self._confidence_series(frame).fillna(1.0) < self.confidence_threshold
        if "annotation_reference_label" in frame.columns and "annotation_auto_label" in frame.columns:
            review_mask = review_mask | (
                frame["annotation_reference_label"].fillna("").astype(str).str.strip()
                != frame["annotation_auto_label"].fillna("").astype(str).str.strip()
            )
        for value in frame.loc[review_mask, signal_column].dropna().astype(str):
            trimmed = value.strip()
            if not trimmed:
                continue
            examples.append(trimmed[:180])
            if len(examples) == 3:
                break
        return examples

    def _write_artifacts(
        self,
        *,
        labeled: pd.DataFrame,
        spec_text: str,
        quality_metrics: Mapping[str, Any],
        labelstudio_payload: Sequence[Mapping[str, Any]],
        review_payload: Sequence[Mapping[str, Any]],
    ) -> AnnotationArtifacts:
        annotated_dataset_path = self.output_dir / "annotated_dataset.jsonl"
        spec_path = self.output_dir / "annotation_spec.md"
        quality_path = self.output_dir / "annotation_quality.json"
        labelstudio_path = self.output_dir / "labelstudio_import.json"
        review_path = self.output_dir / "low_confidence_review.json"
        labeled.to_json(annotated_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        spec_path.write_text(spec_text, encoding="utf-8")
        quality_path.write_text(json.dumps(quality_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        labelstudio_path.write_text(json.dumps(list(labelstudio_payload), indent=2, ensure_ascii=False), encoding="utf-8")
        review_path.write_text(json.dumps(list(review_payload), indent=2, ensure_ascii=False), encoding="utf-8")
        return AnnotationArtifacts(
            annotated_dataset_path=annotated_dataset_path,
            spec_path=spec_path,
            quality_path=quality_path,
            labelstudio_path=labelstudio_path,
            review_path=review_path,
        )

    def _labelstudio_annotation(self, row: pd.Series, signal_column: str) -> dict[str, Any] | None:
        label = self._normalize_optional_text(row.get(self.label_column))
        if label is None:
            return None
        return {
            "id": f"annotation-{row.name}",
            "completed_by": "data_annotation_agent",
            "was_cancelled": False,
            "ground_truth": False,
            "result": [
                {
                    "from_name": "label",
                    "to_name": signal_column,
                    "type": "choices",
                    "value": {"choices": [label]},
                }
            ],
            "lead_time": 0.0,
        }

    def _adapter(self) -> BaseModelAdapter:
        if self._model_adapter is None:
            self._model_adapter = self._build_model_adapter()
        return self._model_adapter

    def _build_model_adapter(self) -> BaseModelAdapter:
        return OllamaAdapter(
            model=str(self.process_config.get("model")),
            base_url=str(self.process_config.get("base_url")),
            api_key=str(self.process_config.get("api_key", "")) or None,
            timeout=int(self.process_config.get("timeout", 60)),
            max_tokens=None if self.process_config.get("max_tokens") is None else int(self.process_config.get("max_tokens")),
            headers=self.process_config.get("headers"),
            max_retries=int(self.process_config.get("request_max_retries", 5)),
            retry_backoff_seconds=float(self.process_config.get("retry_backoff_seconds", 1.0)),
            retry_backoff_max_seconds=float(self.process_config.get("retry_backoff_max_seconds", 30.0)),
        )

    def _chat_json(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        adapter = self._adapter()
        original_timeout = getattr(adapter, "timeout", None)
        if timeout is not None and hasattr(adapter, "timeout"):
            adapter.timeout = int(timeout)
        try:
            response = adapter.chat(messages, json_schema=json_schema)
        finally:
            if timeout is not None and hasattr(adapter, "timeout"):
                adapter.timeout = original_timeout
        if not isinstance(response, dict):
            raise ValueError("Structured LLM response was not a JSON object.")
        return response

    def _effective_parallel_workers(self) -> int:
        requested = max(1, int(self.process_config.get("parallel_workers", 16)))
        if self._uses_remote_backend():
            capped = max(1, int(self.process_config.get("remote_parallel_workers", 2)))
            return min(requested, capped)
        return requested

    def _effective_batch_size(self) -> int:
        configured = self.process_config.get("batch_size")
        if configured is not None:
            return max(1, int(configured))
        if not self._uses_remote_backend():
            return 1
        return max(1, int(self.process_config.get("remote_batch_size", 4)))

    def _uses_remote_backend(self) -> bool:
        base_url = str(self.process_config.get("base_url", "")).strip().lower()
        if not base_url:
            return False
        return not any(host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))

    def _review_reason(self, row: pd.Series, threshold: float) -> str:
        confidence = self._safe_float(row.get("annotation_confidence")) or 0.0
        reference = self._normalize_optional_text(row.get("annotation_reference_label"))
        auto = self._normalize_optional_text(row.get("annotation_auto_label"))
        if reference is not None and auto is not None and reference != auto:
            return "label_disagreement"
        if confidence < threshold:
            return "low_confidence"
        return "manual_review"

    def _row_tokens(self, signal_value: Any, metadata: Any) -> list[str]:
        tokens = self._tokenize(self._stringify_value(signal_value))
        if metadata is not None:
            tokens.extend(self._tokenize(self._stringify_metadata(metadata)))
        return tokens

    def _stringify_metadata(self, metadata: Any) -> str:
        if isinstance(metadata, str):
            return metadata
        try:
            return json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(metadata)

    def _stringify_value(self, value: Any) -> str:
        if self._is_empty(value):
            return ""
        return str(value)

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _cohen_kappa(self, reference: Sequence[str], predicted: Sequence[str]) -> float | None:
        if len(reference) != len(predicted) or len(reference) < 2:
            return None
        labels = sorted(set(reference) | set(predicted))
        observed = sum(1 for ref, pred in zip(reference, predicted, strict=False) if ref == pred) / len(reference)
        expected = 0.0
        ref_counts = Counter(reference)
        pred_counts = Counter(predicted)
        for label in labels:
            expected += (ref_counts[label] / len(reference)) * (pred_counts[label] / len(predicted))
        if math.isclose(1.0 - expected, 0.0):
            return 1.0 if math.isclose(observed, 1.0) else 0.0
        return float((observed - expected) / (1.0 - expected))

    def _normalize_optional_text(self, value: Any) -> str | None:
        if self._is_empty(value):
            return None
        normalized = str(value).strip()
        return normalized or None

    def _safe_float(self, value: Any) -> float | None:
        if self._is_empty(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_bool(self, value: Any) -> bool | None:
        if self._is_empty(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
            return None
        try:
            return bool(value)
        except Exception:
            return None

    def _confidence_series(self, frame: pd.DataFrame) -> pd.Series:
        if "annotation_confidence" not in frame.columns:
            return pd.Series(index=frame.index, dtype=float)
        return pd.to_numeric(frame["annotation_confidence"], errors="coerce")

    def _is_empty(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, float) and math.isnan(value):
            return True
        return bool(pd.isna(value)) if not isinstance(value, (dict, list, tuple, set)) else False

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        raise ValueError(f"Unsupported input dataset format '{path.suffix}'.")

    def _load_config(self, config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _record_log(self, message: str, logs: list[str] | None = None) -> str:
        entry = f"[{datetime.now(UTC).isoformat()}] {message}"
        if logs is not None:
            logs.append(entry)
        print(entry, file=sys.stdout, flush=True)
        return entry

    def _merge_logs(self, existing: list[str], new: list[str]) -> list[str]:
        merged = list(existing)
        for entry in new:
            if entry not in merged:
                merged.append(entry)
        return merged
