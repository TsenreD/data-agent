import json
import math
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from models.ollama_adapter import OllamaAdapter

from ..base import AgentResult, BaseAgent
from .smolagents_backend import SmolagentsAnnotationBackend, SmolagentsParserBackend


DEFAULT_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MAX_TOKENS = 8192
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


@dataclass(slots=True)
class AnnotationRunArtifacts:
    annotated_dataset_path: Path
    spec_path: Path
    quality_path: Path
    labelstudio_path: Path
    review_path: Path


@dataclass(slots=True)
class AnnotationPassResult:
    raw_responses: list[Any]
    logs: list[str] = field(default_factory=list)


class DataAnnotationAgent(BaseAgent):
    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        output_dir: str | Path = "data",
    ) -> None:
        self.config = self._load_config(config)
        self.base_output_dir = Path(output_dir)
        self.output_dir = self._stage_output_dir(self.base_output_dir, "annotation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotation_config = self._resolve_annotation_config()
        self.process_config = self._resolve_process_config()
        self.task = str(self.annotation_config.get("task", "annotation")).strip() or "annotation"
        self.modality = str(
            self.annotation_config.get("modality")
            or self.config.get("project", {}).get("modality")
            or "text"
        ).strip()
        self.label_column = str(self.annotation_config.get("label_column", "label")).strip() or "label"
        self.confidence_threshold = float(
            self.annotation_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        )
        self.user_prompt = str(
            self.annotation_config.get("annotation_prompt") or self.annotation_config.get("prompt") or ""
        ).strip()
        classes = self.annotation_config.get("classes", [])
        self.classes = [dict(item) for item in classes if isinstance(item, Mapping)]

        self.annotation_backend = SmolagentsAnnotationBackend(
            llm_config=self.config.get("llm", {}),
            agent_config=self.annotation_config.get("setup_agent", {}),
        )
        self.parser_backend = SmolagentsParserBackend(
            llm_config=self.config.get("llm", {}),
            agent_config=self.annotation_config.get("parser_agent", {}),
        )

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result = self.execute({"dataframe": dataframe})
        if result.dataframe is None:
            raise RuntimeError("DataAnnotationAgent did not produce a dataframe.")
        return result.dataframe

    def execute(self, payload: Any | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting DataAnnotationAgent execution.", logs)

        upstream = payload if isinstance(payload, AgentResult) else None
        frame = self._resolve_dataframe(payload)
        self._record_log(f"Loaded dataframe with {len(frame)} rows.", logs)

        labeled, annotation_metadata, auto_label_logs = self.auto_label(
            frame,
            prompt=self.user_prompt or None,
            logs=logs,
            return_metadata=True,
        )
        logs.extend(auto_label_logs)

        quality = self.check_quality(labeled)
        review_payload = self.flag_for_review(labeled, threshold=self.confidence_threshold)
        labelstudio_payload = self.export_to_labelstudio(labeled, review_only=False)
        spec_text = self.generate_spec(labeled, self.task)
        artifacts = self._write_artifacts(
            labeled=labeled,
            spec_text=spec_text,
            quality_metrics=quality,
            labelstudio_payload=labelstudio_payload,
            review_payload=review_payload,
        )

        self._record_log(
            f"DataAnnotationAgent execution finished with {quality['review_count']} review rows.",
            logs,
        )

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
                "task": self.task,
                "modality": self.modality,
                "prompt": self.user_prompt or None,
                "confidence_threshold": self.confidence_threshold,
                "selected_columns": annotation_metadata.get("selected_columns", []),
                "quality": quality,
            },
        }
        merged_metrics = {
            **upstream_metrics,
            "row_count": int(len(labeled)),
            "annotation_low_confidence_count": int(quality["review_count"]),
            "annotation_confidence_mean": float(quality["confidence_mean"]),
        }
        merged_logs = self._merge_logs(upstream_logs, logs)

        return AgentResult(
            dataframe=labeled,
            dataframe_path=artifacts.annotated_dataset_path,
            dataframe_schema={column: str(dtype) for column, dtype in labeled.dtypes.items()},
            metrics=merged_metrics,
            artifacts=merged_artifacts,
            logs=merged_logs,
            metadata=merged_metadata,
        )

    def auto_label(
        self,
        dataframe: pd.DataFrame,
        prompt: str | None = None,
        logs: list[str] | None = None,
        *,
        return_metadata: bool = False,
    ) -> Any:
        prompt = (prompt or "").strip()
        work_df = dataframe.copy()
        label_column = self._resolved_label_column(work_df)
        existing_rows = self._existing_label_mask(work_df, label_column)
        labeled, metadata = self._auto_label_standard(work_df, existing_rows, logs, prompt=prompt or None)

        if label_column in dataframe.columns:
            existing_values = dataframe[label_column]
            for index, has_existing in existing_rows.items():
                if bool(has_existing):
                    labeled.at[index, label_column] = existing_values.loc[index]

        if return_metadata:
            return labeled, metadata, []
        return labeled

    def generate_spec(self, df: pd.DataFrame, task: str) -> str:
        label_column = self._resolved_label_column(df)
        lines = [
            "# Annotation Specification",
            "",
            "## Task",
            "",
            f"- Task: `{task}`",
            f"- Objective: Produce `{label_column}` for the configured annotation task.",
            f"- Modality: `{self.modality}`",
            f"- Target column: `{label_column}`",
            f"- Human review threshold: `{self.confidence_threshold}`",
            "",
            "## Classes",
            "",
        ]

        if self.classes:
            for class_config in self.classes:
                name = str(class_config.get("name", "unknown")).strip() or "unknown"
                description = str(class_config.get("description", "No explicit description configured.")).strip()
                keywords = class_config.get("keywords", [])
                lines.extend(
                    [
                        f"### {name}",
                        "",
                        f"- Definition: {description}",
                        f"- Prompt guidance keywords: {', '.join(str(item) for item in keywords) if keywords else 'none'}",
                        "- Examples:",
                    ]
                )
                examples = self._examples_for_label(df, name)
                if not examples:
                    examples = [str(example) for example in class_config.get("examples", [])][:3]
                if not examples:
                    lines.append("  - No examples available yet.")
                else:
                    for example in examples[:3]:
                        lines.append(f"  - {example}")
                lines.extend(["", "#### Edge Cases", "", "- Preserve formatting that changes meaning.", "- Use human review when the answer is ambiguous.", ""])
        else:
            lines.extend(
                [
                    "- No explicit classes configured.",
                    "",
                    "## Edge Cases",
                    "",
                    "- Preserve math, code, or symbolic formatting when present.",
                    "- Leave rows unresolved when the model output is incomplete or not confidently parseable.",
                    "",
                ]
            )

        return "\n".join(lines).strip() + "\n"

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        label_column = self._resolved_label_column(df_labeled)
        final_labels = df_labeled.get("annotation_auto_label", pd.Series([None] * len(df_labeled), index=df_labeled.index))
        pass_1 = df_labeled.get("annotation_auto_label_pass_1", pd.Series([None] * len(df_labeled), index=df_labeled.index))
        pass_2 = df_labeled.get("annotation_auto_label_pass_2", pd.Series([None] * len(df_labeled), index=df_labeled.index))
        intra = self._compute_pairwise_agreement(pass_1, pass_2)

        inter_rate = None
        inter_kappa = None
        if self._supports_inter_expert(df_labeled, label_column):
            reference = df_labeled[label_column].apply(self._normalize_answer)
            predicted = final_labels.apply(self._normalize_answer)
            comparable = pd.DataFrame({"reference": reference, "predicted": predicted}).dropna()
            if not comparable.empty:
                inter_rate = float((comparable["reference"] == comparable["predicted"]).mean())
            if len(comparable) >= 2:
                inter_kappa = self._cohen_kappa(
                    comparable["reference"].astype(str).tolist(),
                    comparable["predicted"].astype(str).tolist(),
                )

        confidence = pd.to_numeric(df_labeled.get("annotation_confidence"), errors="coerce").fillna(0.0)
        label_dist = (
            final_labels.dropna().astype(str).value_counts().to_dict()
            if "annotation_auto_label" in df_labeled.columns
            else {}
        )
        review_count = int(df_labeled.get("annotation_needs_review", pd.Series(dtype=bool)).fillna(False).sum())

        return {
            "kappa": inter_kappa,
            "agreement_pct": None if inter_rate is None else float(inter_rate * 100.0),
            "agreement_rate": inter_rate,
            "inter_expert_kappa": inter_kappa,
            "inter_expert_agreement_pct": None if inter_rate is None else float(inter_rate * 100.0),
            "inter_expert_agreement_rate": inter_rate,
            "intra_agreement_kappa": intra["kappa"],
            "intra_agreement_pct": None if intra["agreement_rate"] is None else float(intra["agreement_rate"] * 100.0),
            "intra_agreement_rate": intra["agreement_rate"],
            "label_dist": label_dist,
            "confidence_mean": float(confidence.mean()) if len(confidence) else 0.0,
            "review_count": review_count,
            "review_rate": 0.0 if len(df_labeled) == 0 else float(review_count / len(df_labeled)),
        }

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        review_only: bool = False,
    ) -> list[dict[str, Any]]:
        if review_only:
            mask = df.get("annotation_needs_review", pd.Series(False, index=df.index)).fillna(False)
            working = df[mask].copy()
        else:
            working = df.copy()

        tasks: list[dict[str, Any]] = []
        label_column = self._resolved_label_column(working)
        for index, (_, row) in enumerate(working.iterrows(), start=1):
            text_value = row.get("text")
            data = {"text": None if text_value is None else str(text_value)}
            if "source" in working.columns:
                data["source"] = self._json_ready(row.get("source"))
            for key in ("annotation_confidence", "annotation_label_origin"):
                if key in working.columns:
                    data[key] = self._json_ready(row.get(key))

            final_label = row.get("annotation_auto_label")
            if final_label is None and label_column in working.columns:
                final_label = row.get(label_column)
            annotation_result = []
            if self._normalize_answer(final_label) is not None:
                annotation_result.append(
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": [str(final_label)]},
                    }
                )

            tasks.append(
                {
                    "id": index,
                    "data": data,
                    "meta": {
                        "needs_review": bool(row.get("annotation_needs_review", False)),
                        "review_reason": self._review_reason(row),
                        "confidence": float(row.get("annotation_confidence", 0.0) or 0.0),
                        "source": self._json_ready(row.get("source")),
                    },
                    "annotations": [
                        {
                            "id": f"annotation-{index - 1}",
                            "completed_by": "data_annotation_agent",
                            "was_cancelled": False,
                            "ground_truth": False,
                            "result": annotation_result,
                            "lead_time": 0.0,
                        }
                    ],
                }
            )
        return tasks

    def flag_for_review(self, df_labeled: pd.DataFrame, threshold: float | None = None) -> list[dict[str, Any]]:
        threshold = self.confidence_threshold if threshold is None else float(threshold)
        review_mask = (
            df_labeled.get("annotation_needs_review", pd.Series(False, index=df_labeled.index)).fillna(False)
            | (pd.to_numeric(df_labeled.get("annotation_confidence"), errors="coerce").fillna(0.0) < threshold)
        )
        review_df = df_labeled[review_mask].copy()
        return self.export_to_labelstudio(review_df, review_only=False)

    def process(self, df_path: str | Path, prompt: str) -> Path:
        path = Path(df_path)
        dataframe = self._read_dataframe(path)
        records = [self._json_ready(record) for record in dataframe.to_dict(orient="records")]
        max_workers = max(1, int(self.process_config.get("parallel_workers", 1)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._transform_single_row, row=record, prompt=prompt): index
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    new_row = future.result()
                except Exception:
                    new_row = None
                if not isinstance(new_row, Mapping):
                    continue
                for key, value in new_row.items():
                    dataframe.at[index, key] = value

        output_path = path.with_name(f"{path.stem}_processed{path.suffix}")
        self._write_dataframe(dataframe, output_path)
        return output_path

    def _auto_label_standard(
        self,
        work_df: pd.DataFrame,
        existing_rows: pd.Series,
        logs: list[str] | None,
        *,
        prompt: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        effective_prompt = (prompt or self._standard_annotation_prompt()).strip()
        selected_columns = self._default_selected_columns(work_df)
        selected_columns, fewshot_samples = self._select_columns_with_fewshot(
            work_df,
            effective_prompt,
            selected_columns=selected_columns,
            logs=logs,
        )
        if logs is not None:
            self._record_log(f"Selected columns for annotation: {selected_columns}", logs)
            self._record_log(
                f"Few-shot sample count: {len(fewshot_samples)}",
                logs,
            )
            rows_to_annotate_count = int((~existing_rows).sum())
            self._record_log(
                f"Rows requiring annotation (missing labels): {rows_to_annotate_count} / {len(work_df)}",
                logs,
            )

        selected_payloads = self._build_selected_payloads(work_df, selected_columns)
        missing_indices = [
            index
            for index, has_existing in enumerate(existing_rows.tolist())
            if not bool(has_existing)
        ]
        selected_payloads_missing = [selected_payloads[index] for index in missing_indices]

        pass_1_missing = self._run_annotation_pass(
            rows=selected_payloads_missing,
            pass_name="pass_1",
            prompt=effective_prompt,
            fewshot_samples=fewshot_samples,
        )
        pass_2_missing = self._run_annotation_pass(
            rows=selected_payloads_missing,
            pass_name="pass_2",
            prompt=effective_prompt,
            fewshot_samples=fewshot_samples,
        )
        pass_1_full_raw: list[Any] = [None] * len(work_df)
        pass_2_full_raw: list[Any] = [None] * len(work_df)
        for offset, row_index in enumerate(missing_indices):
            if offset < len(pass_1_missing.raw_responses):
                pass_1_full_raw[row_index] = pass_1_missing.raw_responses[offset]
            if offset < len(pass_2_missing.raw_responses):
                pass_2_full_raw[row_index] = pass_2_missing.raw_responses[offset]

        pass_1 = AnnotationPassResult(raw_responses=pass_1_full_raw, logs=pass_1_missing.logs)
        pass_2 = AnnotationPassResult(raw_responses=pass_2_full_raw, logs=pass_2_missing.logs)

        work_df["annotation_raw_response_1"] = [self._serialize_raw_output(item) for item in pass_1.raw_responses]
        work_df["annotation_raw_response_2"] = [self._serialize_raw_output(item) for item in pass_2.raw_responses]
        staged_path = self.output_dir / "_annotation_parser_input.jsonl"
        work_df.to_json(staged_path, orient="records", lines=True, force_ascii=False, date_format="iso")

        sample_rows = [
            self._json_ready(record)
            for record in work_df.head(min(5, len(work_df))).to_dict(orient="records")
        ]
        parsed_1 = self.parser_backend.parse_answers(
            staged_path,
            column_name="annotation_raw_response_1",
            task_prompt=effective_prompt,
            selected_columns=selected_columns,
            sample_rows=sample_rows,
        )
        parsed_2 = self.parser_backend.parse_answers(
            staged_path,
            column_name="annotation_raw_response_2",
            task_prompt=effective_prompt,
            selected_columns=selected_columns,
            sample_rows=sample_rows,
        )

        work_df["annotation_extracted_answer_1"] = parsed_1.get("answers", [None] * len(work_df))
        work_df["annotation_extracted_answer_2"] = parsed_2.get("answers", [None] * len(work_df))

        existing_label_values = work_df[self._resolved_label_column(work_df)].copy()
        finalized = self._finalize_annotations(
            work_df,
            existing_rows=existing_rows,
            existing_label_values=existing_label_values,
            logs=logs,
        )
        return finalized, {"selected_columns": selected_columns, "fewshot_samples": fewshot_samples}

    def _run_annotation_pass(
        self,
        *,
        rows: list[dict[str, Any]],
        pass_name: str,
        prompt: str,
        fewshot_samples: Sequence[Mapping[str, Any]],
    ) -> AnnotationPassResult:
        logs: list[str] = []
        self._record_log(f"Starting annotation {pass_name}.", logs)
        if rows:
            sample_payload = {
                "task": self.task,
                "prompt": prompt,
                "classes": self.classes,
                "examples": list(fewshot_samples),
                **rows[0],
            }
            self._record_log(
                "Annotation payload sample for "
                f"{pass_name}: {json.dumps(sample_payload, ensure_ascii=False)}",
                logs,
            )
        raw_responses: list[Any] = [None] * len(rows)
        max_workers = max(1, int(self.process_config.get("parallel_workers", 1)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._annotate_single_row,
                    row_index=index,
                    row=row,
                    prompt=prompt,
                    fewshot_samples=fewshot_samples,
                ): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    raw_responses[index] = future.result()
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    self._record_log(f"Annotation {pass_name} row {index} error: {message}", logs)
                    raw_responses[index] = None
        self._record_log(f"Finished annotation {pass_name}.", logs)
        return AnnotationPassResult(raw_responses=raw_responses, logs=logs)

    def _annotate_single_row(
        self,
        *,
        row_index: int,
        row: Mapping[str, Any],
        prompt: str,
        fewshot_samples: Sequence[Mapping[str, Any]],
    ) -> str:
        adapter = self._build_model_adapter()
        row_payload = {
            "task": self.task,
            "prompt": prompt,
            "classes": self.classes,
            "examples": list(fewshot_samples),
            **row,
        }
        return self._annotation_chat(adapter, row_payload, row_index=row_index)

    def _select_columns_with_fewshot(
        self,
        df: pd.DataFrame,
        prompt: str,
        *,
        selected_columns: Sequence[str],
        logs: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            columns, fewshot_samples = self.annotation_backend.select_columns_with_fewshot(
                user_prompt=prompt,
                columns=[str(column) for column in selected_columns],
            )
        except Exception as error:
            if logs is not None:
                self._record_log(
                    f"Annotation setup backend failed; falling back to host heuristics: {type(error).__name__}: {error}",
                    logs,
                )
            columns, fewshot_samples = [], []

        selected = [column for column in columns if column in df.columns]
        if not selected:
            selected = [str(column) for column in selected_columns if column in df.columns]
        if not selected:
            selected = self._default_selected_columns(df)
        examples = [dict(item) for item in fewshot_samples if isinstance(item, Mapping)]
        return selected, examples

    def _default_selected_columns(self, df: pd.DataFrame) -> list[str]:
        preferred = [column for column in ["text", "problem", "question", "prompt"] if column in df.columns]
        if preferred:
            return preferred
        return [str(column) for column in df.columns]

    def _build_selected_payloads(self, df: pd.DataFrame, selected_columns: Sequence[str]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            payloads.append({column: self._json_ready(row.get(column)) for column in selected_columns})
        return payloads

    def _finalize_annotations(
        self,
        df: pd.DataFrame,
        *,
        existing_rows: pd.Series,
        existing_label_values: pd.Series,
        logs: list[str] | None,
    ) -> pd.DataFrame:
        label_column = self._resolved_label_column(df)
        answer_1 = df["annotation_extracted_answer_1"].apply(self._normalize_answer)
        answer_2 = df["annotation_extracted_answer_2"].apply(self._normalize_answer)

        auto_labels: list[str | None] = []
        label_origins: list[str] = []
        intra_agreement: list[bool] = []
        confidence: list[float] = []
        review_flags: list[bool] = []
        compare_statuses: list[str] = []
        compare_errors: list[str | None] = []

        pass_1_labels: list[str | None] = []
        pass_2_labels: list[str | None] = []
        agreement_decisions = self._llm_compare_pass_answers(
            answer_1=answer_1.tolist(),
            answer_2=answer_2.tolist(),
            compare_mask=[not bool(item) for item in existing_rows.tolist()],
            logs=logs,
        )

        for idx, (a1, a2) in enumerate(zip(answer_1.tolist(), answer_2.tolist(), strict=False)):
            pass_1_labels.append(a1)
            pass_2_labels.append(a2)
            agreement_decision = agreement_decisions[idx] if idx < len(agreement_decisions) else {}
            compare_status = str(agreement_decision.get("status", "error"))
            compare_error = (
                str(agreement_decision.get("error"))
                if agreement_decision.get("error") is not None
                else None
            )
            llm_agree = bool(agreement_decision.get("agree", False))
            llm_final_label = self._normalize_answer(agreement_decision.get("final_label"))
            compare_failed = compare_status == "error"

            if bool(existing_rows.iloc[idx]):
                final_label = self._normalize_answer(existing_label_values.iloc[idx])
                label_origin = "existing"
                row_confidence = 1.0 if not compare_failed else 0.0
            elif compare_failed:
                final_label = None
                label_origin = "unresolved"
                row_confidence = 0.0
            elif llm_agree:
                final_label = llm_final_label or a1 or a2
                label_origin = "agreed_llm" if final_label is not None else "unresolved"
                row_confidence = 1.0 if final_label is not None else 0.0
            else:
                final_label = None
                label_origin = "unresolved"
                row_confidence = 0.0

            agreed = llm_agree
            needs_review = compare_failed or (not agreed) or final_label is None

            auto_labels.append(final_label)
            label_origins.append(label_origin)
            intra_agreement.append(bool(agreed))
            confidence.append(float(row_confidence))
            review_flags.append(bool(needs_review))
            compare_statuses.append(compare_status)
            compare_errors.append(compare_error)

        df["annotation_auto_label_pass_1"] = pass_1_labels
        df["annotation_auto_label_pass_2"] = pass_2_labels
        df["annotation_auto_label"] = auto_labels
        df["annotation_label_origin"] = label_origins
        df["annotation_intra_agreement"] = intra_agreement
        df["annotation_confidence"] = confidence
        df["annotation_needs_review"] = review_flags
        df["annotation_compare_status"] = compare_statuses
        df["annotation_compare_error"] = compare_errors

        for idx, final_label in enumerate(auto_labels):
            if final_label is not None and not bool(existing_rows.iloc[idx]):
                df.at[df.index[idx], label_column] = final_label
        return df

    def _llm_compare_pass_answers(
        self,
        *,
        answer_1: Sequence[str | None],
        answer_2: Sequence[str | None],
        compare_mask: Sequence[bool],
        logs: list[str] | None,
    ) -> list[dict[str, Any]]:
        if logs is not None:
            self._record_log("Starting answer reconciliation.", logs)
        max_workers = max(1, int(self.process_config.get("parallel_workers", 1)))
        decisions: list[dict[str, Any]] = [
            {"status": "skipped", "agree": True, "final_label": None, "error": None}
            for _ in range(len(answer_1))
        ]

        compare_targets: list[tuple[int, str | None, str | None]] = []
        for index, (first, second) in enumerate(zip(answer_1, answer_2, strict=False)):
            if index >= len(compare_mask) or not bool(compare_mask[index]):
                continue
            normalized_first = self._normalize_answer(first)
            normalized_second = self._normalize_answer(second)
            if normalized_first == normalized_second:
                decisions[index] = {
                    "status": "ok",
                    "agree": True,
                    "final_label": normalized_first,
                    "error": None,
                }
                continue
            compare_targets.append((index, normalized_first, normalized_second))

        if compare_targets and logs is not None:
            sample_index, sample_first, sample_second = compare_targets[0]
            sample_payload = {
                "task": self.task,
                "user_prompt": self._standard_annotation_prompt(),
                "pass_1_answer": sample_first,
                "pass_2_answer": sample_second,
                "row_index": sample_index,
                "instruction": (
                    "Compare only the final label semantics of the two answers. Ignore explanation style, reasoning "
                    "chain, or formatting differences that do not change the final label. Return JSON with `agree` "
                    "and `final_label`. If both answers imply the same final label, set `agree` to true and return "
                    "that canonical final label."
                ),
            }
            self._record_log(
                "Answer reconciliation payload sample: "
                f"{json.dumps(sample_payload, ensure_ascii=False)}",
                logs,
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._compare_single_pass_pair,
                    pass_1=first,
                    pass_2=second,
                ): index
                for index, first, second in compare_targets
            }
            for future in as_completed(futures):
                index = futures[future]
                decision = future.result()
                if logs is not None and isinstance(decision, Mapping) and decision.get("status") == "error":
                    self._record_log(
                        f"Answer reconciliation row {index} error: {decision.get('error')}",
                        logs,
                    )
                decisions[index] = decision
        if logs is not None:
            self._record_log("Finished answer reconciliation.", logs)
        return decisions

    def _compare_single_pass_pair(
        self,
        *,
        pass_1: str | None,
        pass_2: str | None,
    ) -> dict[str, Any]:
        adapter = self._build_model_adapter()
        payload = {
            "task": self.task,
            "user_prompt": self._standard_annotation_prompt(),
            "pass_1_answer": pass_1,
            "pass_2_answer": pass_2,
            "instruction": (
                "Compare only the final label semantics of the two answers. Ignore explanation style, reasoning chain, "
                "or formatting differences that do not change the final label. Return JSON with `agree` and "
                "`final_label`. If both answers imply the same final label, set `agree` to true and return that "
                "canonical final label."
            ),
        }
        messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        try:
            raw_response = adapter.chat(messages, json_schema=None)
            return self._coerce_compare_response(
                raw_response,
                pass_1=pass_1,
                pass_2=pass_2,
            )
        except Exception as raw_error:
            message = f"compare call failed: {type(raw_error).__name__}: {raw_error}"
            return {"status": "error", "agree": False, "final_label": None, "error": message}

    def _coerce_compare_response(
        self,
        raw_response: Any,
        *,
        pass_1: str | None,
        pass_2: str | None,
    ) -> dict[str, Any]:
        if isinstance(raw_response, Mapping):
            agree = bool(raw_response.get("agree", False))
            return {
                "status": "ok",
                "agree": agree,
                "final_label": self._normalize_answer(raw_response.get("final_label")),
                "error": None,
            }

        text = "" if raw_response is None else str(raw_response).strip()
        if text:
            candidates = [text]
            match = JSON_FENCE_PATTERN.search(text)
            if match:
                candidates.append(match.group(1).strip())
            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping):
                    agree = bool(parsed.get("agree", False))
                    return {
                        "status": "ok",
                        "agree": agree,
                        "final_label": self._normalize_answer(parsed.get("final_label")),
                        "error": None,
                    }

        normalized_1 = self._normalize_answer(pass_1)
        normalized_2 = self._normalize_answer(pass_2)
        agree = normalized_1 is not None and normalized_1 == normalized_2
        return {
            "status": "ok",
            "agree": agree,
            "final_label": normalized_1 if agree else None,
            "error": None,
        }

    def _standard_annotation_prompt(self) -> str:
        if self.user_prompt:
            return self.user_prompt
        label_guidance = ""
        if self.classes:
            rendered = []
            for class_config in self.classes:
                name = str(class_config.get("name", "")).strip()
                if not name:
                    continue
                description = str(class_config.get("description", "")).strip()
                rendered.append(f"- {name}: {description or 'use this label when appropriate'}")
            if rendered:
                label_guidance = "\nConfigured labels:\n" + "\n".join(rendered)
        return (
            f"Annotate the input rows for task `{self.task}`.{label_guidance}\n"
            "Return the answer in plain text only. Do not wrap it in JSON."
        )

    def _build_model_adapter(self):
        return OllamaAdapter(
            model=str(self.process_config.get("model", "default")),
            base_url=str(self.process_config["base_url"]),
            api_key=self.process_config.get("api_key"),
            timeout=int(self.process_config.get("timeout_per_row", 120)),
            max_tokens=int(self.process_config.get("max_tokens", DEFAULT_MAX_TOKENS)),
            headers=self.process_config.get("headers"),
            max_retries=int(self.process_config.get("max_retries", 1)),
        )

    def _annotation_chat(
        self,
        adapter: Any,
        payload: Mapping[str, Any],
        *,
        row_index: int,
    ) -> str:
        try:
            raw_response = adapter.chat(
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                json_schema=None,
            )
            if raw_response is None:
                return ""
            return str(raw_response).strip()
        except Exception as raw_error:
            raise RuntimeError(
                f"annotation call failed for row {row_index}: {type(raw_error).__name__}: {raw_error}"
            ) from raw_error

    def _transform_single_row(
        self,
        *,
        row: Mapping[str, Any],
        prompt: str,
    ) -> dict[str, Any] | None:
        adapter = self._build_model_adapter()
        raw_response = adapter.chat(
            [
                {
                    "role": "user",
                    "content": json.dumps({"prompt": prompt, **row}, ensure_ascii=False),
                }
            ],
            json_schema=None,
        )
        if isinstance(raw_response, Mapping):
            return dict(raw_response)
        text = "" if raw_response is None else str(raw_response).strip()
        if not text:
            return None
        candidates = [text]
        match = JSON_FENCE_PATTERN.search(text)
        if match:
            candidates.append(match.group(1).strip())
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    def _examples_for_label(self, df: pd.DataFrame, label_value: str) -> list[str]:
        if "annotation_auto_label" in df.columns:
            subset = df[df["annotation_auto_label"].astype(str) == label_value]
        elif self.label_column in df.columns:
            subset = df[df[self.label_column].astype(str) == label_value]
        else:
            subset = df.iloc[0:0]
        if subset.empty:
            return []
        source_column = "text" if "text" in subset.columns else subset.columns[0]
        return [str(value) for value in subset[source_column].dropna().astype(str).head(3).tolist()]

    def _compute_pairwise_agreement(self, first: pd.Series, second: pd.Series) -> dict[str, Any]:
        normalized_first = first.apply(self._normalize_answer)
        normalized_second = second.apply(self._normalize_answer)
        comparable = pd.DataFrame({"first": normalized_first, "second": normalized_second}).dropna()
        agreement_rate = None
        kappa = None
        if not comparable.empty:
            agreement_rate = float((comparable["first"] == comparable["second"]).mean())
        if len(comparable) >= 2:
            kappa = self._cohen_kappa(
                comparable["first"].astype(str).tolist(),
                comparable["second"].astype(str).tolist(),
            )
        return {"agreement_rate": agreement_rate, "kappa": kappa}

    def _supports_inter_expert(self, df: pd.DataFrame, label_column: str) -> bool:
        if label_column not in df.columns:
            return False
        label_series = df[label_column].dropna().astype(str)
        if len(label_series) < 2:
            return False
        if label_series.nunique() > min(20, max(5, len(label_series) // 2)):
            return False
        numeric_ratio = sum(self._looks_numeric(value) for value in label_series.tolist()) / max(1, len(label_series))
        return numeric_ratio < 0.8

    def _resolved_label_column(self, df: pd.DataFrame) -> str:
        if self.label_column in df.columns:
            return self.label_column
        if "label" in df.columns:
            return "label"
        df[self.label_column] = None
        return self.label_column

    def _existing_label_mask(self, df: pd.DataFrame, label_column: str) -> pd.Series:
        if label_column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[label_column].apply(lambda value: self._normalize_answer(value) is not None)

    def _write_artifacts(
        self,
        *,
        labeled: pd.DataFrame,
        spec_text: str,
        quality_metrics: Mapping[str, Any],
        labelstudio_payload: Sequence[Mapping[str, Any]],
        review_payload: Sequence[Mapping[str, Any]],
    ) -> AnnotationRunArtifacts:
        annotated_dataset_path = self.output_dir / "annotated_dataset.jsonl"
        spec_path = self.output_dir / "annotation_spec.md"
        quality_path = self.output_dir / "annotation_quality.json"
        labelstudio_path = self.output_dir / "labelstudio_import.json"
        review_path = self.output_dir / "low_confidence_review.json"

        labeled.to_json(annotated_dataset_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        spec_path.write_text(spec_text, encoding="utf-8")
        quality_path.write_text(json.dumps(dict(quality_metrics), indent=2, ensure_ascii=False), encoding="utf-8")
        labelstudio_path.write_text(json.dumps(list(labelstudio_payload), indent=2, ensure_ascii=False), encoding="utf-8")
        review_path.write_text(json.dumps(list(review_payload), indent=2, ensure_ascii=False), encoding="utf-8")

        return AnnotationRunArtifacts(
            annotated_dataset_path=annotated_dataset_path,
            spec_path=spec_path,
            quality_path=quality_path,
            labelstudio_path=labelstudio_path,
            review_path=review_path,
        )

    def _review_reason(self, row: pd.Series) -> str:
        if bool(row.get("annotation_needs_review")):
            if str(row.get("annotation_compare_status", "ok")) != "ok":
                return "reconciliation_error"
            a1 = self._normalize_answer(row.get("annotation_extracted_answer_1"))
            a2 = self._normalize_answer(row.get("annotation_extracted_answer_2"))
            if a1 is None or a2 is None:
                return "parser_failed_or_empty_answer"
            if a1 != a2:
                return "pass_disagreement"
            if float(row.get("annotation_confidence", 0.0) or 0.0) < self.confidence_threshold:
                return "low_confidence"
            return "unresolved_row"
        return "none"

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
        quality_input_path = self._stage_output_dir(self.base_output_dir, "quality") / "cleaned_dataset.jsonl"
        if quality_input_path.exists():
            return self._read_dataframe(quality_input_path)
        collection_input_path = self._stage_output_dir(self.base_output_dir, "collection") / "unified_dataset.jsonl"
        if collection_input_path.exists():
            return self._read_dataframe(collection_input_path)
        raise ValueError("DataAnnotationAgent requires payload['dataframe'] or annotation.input_path.")

    def _resolve_annotation_config(self) -> dict[str, Any]:
        agents_config = self.config.get("agents", {})
        if isinstance(agents_config, Mapping) and isinstance(agents_config.get("annotation"), Mapping):
            return dict(agents_config["annotation"])
        annotation_config = self.config.get("annotation", {})
        if isinstance(annotation_config, Mapping):
            return dict(annotation_config)
        return {}

    def _resolve_process_config(self) -> dict[str, Any]:
        llm_config = self.config.get("llm", {})
        configured = self.annotation_config.get("process_config", {})
        if not isinstance(configured, Mapping):
            configured = {}
        merged = {
            "parallel_workers": 200,
            "timeout_per_row": 120,
            "max_retries": 1,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "model": configured.get("model") or llm_config.get("model") or "default",
            "base_url": self.annotation_config.get("base_url")
            or configured.get("base_url")
            or llm_config.get("base_url")
            or "http://localhost:11434/v1/chat/completions",
            "api_key": configured.get("api_key") or self.annotation_config.get("api_key") or llm_config.get("api_key"),
            "headers": dict(configured.get("headers", {})),
        }
        merged.update(dict(configured))
        merged["base_url"] = (
            self.annotation_config.get("base_url")
            or merged.get("base_url")
            or "http://localhost:11434/v1/chat/completions"
        )
        merged["max_tokens"] = int(merged.get("max_tokens", DEFAULT_MAX_TOKENS))
        merged["parallel_workers"] = self._effective_parallel_workers(int(merged.get("parallel_workers", 200)))
        return merged

    def _effective_parallel_workers(self, configured_workers: int | None = None) -> int:
        return max(1, int(configured_workers or 200))

    def _write_dataframe(self, dataframe: pd.DataFrame, path: Path) -> None:
        if path.suffix.lower() == ".csv":
            dataframe.to_csv(path, index=False)
            return
        dataframe.to_json(path, orient="records", lines=path.suffix.lower() == ".jsonl", force_ascii=False, date_format="iso")

    def _normalize_answer(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None
        return " ".join(text.split())

    def _looks_numeric(self, value: str) -> bool:
        text = value.strip()
        if not text:
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _serialize_raw_output(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _cohen_kappa(self, reference: Sequence[str], predicted: Sequence[str]) -> float | None:
        if len(reference) != len(predicted) or len(reference) < 2:
            return None
        labels = sorted(set(reference) | set(predicted))
        observed = sum(1 for ref, pred in zip(reference, predicted, strict=False) if ref == pred) / len(reference)
        ref_counts = {label: reference.count(label) for label in labels}
        pred_counts = {label: predicted.count(label) for label in labels}
        expected = 0.0
        for label in labels:
            expected += (ref_counts[label] / len(reference)) * (pred_counts[label] / len(predicted))
        if math.isclose(1.0 - expected, 0.0):
            return 1.0 if math.isclose(observed, 1.0) else 0.0
        return float((observed - expected) / (1.0 - expected))

    def _json_ready(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_ready(item) for item in value]
        return str(value)

    def _read_dataframe(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        raise ValueError(f"Unsupported dataframe format: {path.suffix}")

    def _stage_output_dir(self, output_dir: Path, stage_name: str) -> Path:
        if output_dir.name == stage_name:
            return output_dir
        return output_dir / stage_name

    def _load_config(self, config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(config, Mapping):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _merge_logs(self, upstream_logs: Sequence[str], current_logs: Sequence[str]) -> list[str]:
        merged: list[str] = []
        for entry in [*upstream_logs, *current_logs]:
            if entry not in merged:
                merged.append(entry)
        return merged

    def _record_log(self, message: str, logs: list[str] | None = None) -> str:
        entry = f"[{datetime.now(UTC).isoformat()}] {message}"
        if logs is not None:
            logs.append(entry)
        print(entry, file=sys.stdout, flush=True)
        return entry
