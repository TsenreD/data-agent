import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from ..base import AgentResult, BaseAgent
from .smolagents_backend import SmolagentsAnnotationBackend, SmolagentsParserBackend


@dataclass(slots=True)
class AnnotationRunArtifacts:
    annotated_dataset_path: Path
    review_path: Path
    labelstudio_path: Path
    quality_path: Path
    selected_columns_path: Path


@dataclass(slots=True)
class AnnotationPassResult:
    raw_responses: list[str | None]
    prompt_payloads: list[dict[str, Any]]
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

        self.base_url = str(self.annotation_config["base_url"])
        self.user_prompt = str(self.annotation_config["prompt"]).strip()

        self.annotation_backend = SmolagentsAnnotationBackend(
            base_url=self.base_url,
            model=str(self.process_config.get("model", "default")),
            timeout=int(self.process_config.get("timeout_per_row", 120)),
            max_tokens=int(self.process_config.get("max_tokens", 8192)),
            parallel_workers=int(self.process_config.get("parallel_workers", 200)),
            max_retries=int(self.process_config.get("max_retries", 1)),
            api_key=self.process_config.get("api_key"),
            headers=self.process_config.get("headers"),
        )
        self.parser_backend = SmolagentsParserBackend(
            output_dir=self.output_dir,
            parser_agent_config=self.annotation_config.get("parser_agent", {}),
            llm_config=self.config.get("llm", {}),
        )

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result = self.execute({"dataframe": dataframe})
        if result.dataframe is None:
            raise RuntimeError("DataAnnotationAgent did not produce a dataframe.")
        return result.dataframe
    
    # TODO:
    #     (Cohen's κ, label distribution, confidence)
    #     move from execute function then reuse
    # def check_quality(df_labeled) → QualityMetrics
    #   pass
    
        
    # TODO:"JSON in the  LabelStudio import format"
    #     move from execute function then reuse
    # def export_to_labelstudio(df)
    #     pass 
    
    def auto_label(self, dataframe: pd.DataFrame, prompt: str | None = None, logs: list | None = None) -> pd.DataFrame:
        
        if not prompt:
            prompt = self.user_prompt
        
        # TODO: backend execution, includes prompt from yaml config (check auto_label skill).
        selected_columns, fewshot_samples = self._select_columns_with_fewshot(dataframe)
        if logs:
            self._record_log(f"Selected columns for annotation: {selected_columns}", logs)
            self._record_log(f"Fewshot samples: {fewshot_samples}", logs)

        work_df = dataframe.copy()
        work_df["annotation_selected_payload"] = self._build_selected_payloads(work_df, selected_columns)

        pass_1 = self._run_annotation_pass(work_df, pass_name="pass_1", fewshot_samples)
        pass_2 = self._run_annotation_pass(work_df, pass_name="pass_2", fewshot_samples)

        work_df["annotation_prompt_payload"] = pass_1.prompt_payloads
        work_df["annotation_raw_response_1"] = pass_1.raw_responses
        work_df["annotation_raw_response_2"] = pass_2.raw_responses
        
        # TODO: save work_df so code agent can read it in it's docker

        parsed_1 = self.parser_backend.parse_answers(
            df_path=work_df_path,
            column_name="annotation_raw_response_1",
        )
        parsed_2 = self.parser_backend.parse_answers(
            df_path=work_df_path,
            column_name="annotation_raw_response_2",
        )

        work_df["annotation_extracted_answer_1"] = parsed_1["answers"]
        work_df["annotation_extracted_answer_2"] = parsed_2["answers"]
        
        return work_df

    def execute(self, payload: Mapping[str, Any] | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting DataAnnotationAgent execution.", logs)

        frame = self._resolve_dataframe(payload)
        self._record_log(f"Loaded dataframe with {len(frame)} rows.", logs)

        work_df = self.auto_label(frame, prompt=self.user_prompt, logs=logs)

        agreement = self._compute_intra_expert_agreement(
            work_df["annotation_extracted_answer_1"],
            work_df["annotation_extracted_answer_2"],
        )
        work_df["annotation_intra_agreement"] = agreement["per_row_agreement"]
        work_df["annotation_final_answer"] = agreement["final_answers"]
        work_df["annotation_needs_review"] = agreement["needs_review"]

        quality = self._build_quality_report(work_df, agreement)
        review_rows = work_df[work_df["annotation_needs_review"]].copy()

        labelstudio_payload = self._export_to_labelstudio(review_rows, selected_columns)

        artifacts = self._write_artifacts(
            labeled=work_df,
            review_rows=review_rows,
            labelstudio_payload=labelstudio_payload,
            quality_metrics=quality,
            selected_columns=selected_columns,
        )

        self._record_log(f"Review queue size: {len(review_rows)}", logs)
        self._record_log("DataAnnotationAgent execution finished successfully.", logs)

        return AgentResult(
            dataframe=work_df,
            dataframe_path=artifacts.annotated_dataset_path,
            dataframe_schema={column: str(dtype) for column, dtype in work_df.dtypes.items()},
            metrics={
                "row_count": int(len(work_df)),
                "review_count": int(len(review_rows)),
                "intra_agreement_rate": quality["intra_agreement_rate"],
                "intra_agreement_pct": quality["intra_agreement_pct"],
            },
            artifacts={
                "annotated_dataset": str(artifacts.annotated_dataset_path),
                "review_rows": str(artifacts.review_path),
                "labelstudio_export": str(artifacts.labelstudio_path),
                "quality_report": str(artifacts.quality_path),
                "selected_columns": str(artifacts.selected_columns_path),
            },
            logs=logs + pass_1.logs + pass_2.logs,
            metadata={
                "annotation": {
                    "task": self.task,
                    "confidence_threshold": self.confidence_threshold,
                    "selected_columns": selected_columns,
                }
            },
        )

    def _select_columns_with_fewshot(self, df: pd.DataFrame) -> list[str]:
        sample_rows = df.head(min(8, len(df))).to_dict(orient="records")
        columns, fewshot_samples = self.annotation_backend.select_columns_with_fewshot(
            user_prompt=self.user_prompt,
            columns=[str(c) for c in df.columns],
            sample_rows=sample_rows,
        )
        selected = [c for c in columns if c in df.columns]

        if not selected:
            selected = [c for c in ["text", "problem", "question", "prompt"] if c in df.columns]
        if not selected:
            selected = list(df.columns)

        return selected, fewshot_samples

    def _build_selected_payloads(self, df: pd.DataFrame, selected_columns: Sequence[str]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            payload = {}
            for col in selected_columns:
                payload[col] = self._json_ready(row.get(col))
            payloads.append(payload)
        return payloads

    def _run_annotation_pass(self, df: pd.DataFrame, pass_name: str, fewshot_samples) -> AnnotationPassResult:
        logs: list[str] = []
        self._record_log(f"Starting annotation {pass_name}.", logs)

        prompts = []
        for row_payload in df["annotation_selected_payload"].tolist():
            prompts.append(
                {
                    "user_prompt": self.user_prompt,
                    "fewshot_samples": fewshot_samples,
                    "subset_of_cols": row_payload,
                }
            )

        raw_responses = self.annotation_backend.annotate_batch(prompts)
        self._record_log(f"Finished annotation {pass_name}.", logs)

        return AnnotationPassResult(
            raw_responses=raw_responses,
            prompt_payloads=prompts,
            logs=logs,
        )

    def _generate_parser_code(self, df: pd.DataFrame) -> str:
        samples = []
        for _, row in df.head(min(40, len(df))).iterrows():
            samples.append(
                {
                    "raw_response_1": row.get("annotation_raw_response_1"),
                    "raw_response_2": row.get("annotation_raw_response_2"),
                }
            )
        return self.parser_backend.generate_parser_code(
            task_prompt=self.user_prompt,
            samples=samples,
            parser_spec=(
                "Write a Python parser with function extract_answer(text: str | None) -> str | None. "
                "It should extract the answer inside \\boxed{} when present. "
                "If output indicates incomplete problem or invalid task, return 'invalid'. "
                "Normalize whitespace. Return None on truly unparseable content."
            ),
        )

    def _compute_intra_expert_agreement(
        self,
        answers_1: pd.Series,
        answers_2: pd.Series,
    ) -> dict[str, Any]:
        norm_1 = answers_1.apply(self._normalize_answer)
        norm_2 = answers_2.apply(self._normalize_answer)

        per_row = []
        final_answers = []
        needs_review = []

        for a1, a2 in zip(norm_1.tolist(), norm_2.tolist(), strict=False):
            agree = a1 == a2 and a1 is not None
            per_row.append(bool(agree))
            final_answers.append(a1 if agree else None)
            needs_review.append(not agree)

        comparable = pd.DataFrame({"a1": norm_1, "a2": norm_2}).dropna()
        agreement_rate = None
        kappa = None
        if len(comparable) >= 2:
            agreement_rate = float((comparable["a1"] == comparable["a2"]).mean())
            kappa = self._cohen_kappa(
                comparable["a1"].astype(str).tolist(),
                comparable["a2"].astype(str).tolist(),
            )

        return {
            "per_row_agreement": per_row,
            "final_answers": final_answers,
            "needs_review": needs_review,
            "agreement_rate": agreement_rate,
            "kappa": kappa,
        }

    def _build_quality_report(self, df: pd.DataFrame, agreement: Mapping[str, Any]) -> dict[str, Any]:
        review_count = int(df["annotation_needs_review"].fillna(False).sum())
        return {
            "row_count": int(len(df)),
            "review_count": review_count,
            "review_rate": 0.0 if len(df) == 0 else float(review_count / len(df)),
            "intra_agreement_rate": agreement["agreement_rate"],
            "intra_agreement_pct": None if agreement["agreement_rate"] is None else float(agreement["agreement_rate"] * 100.0),
            "intra_agreement_kappa": agreement["kappa"],
        }

    def _export_to_labelstudio(
        self,
        df: pd.DataFrame,
        selected_columns: Sequence[str],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            data = {col: self._json_ready(row.get(col)) for col in selected_columns}
            data["annotation_raw_response_1"] = row.get("annotation_raw_response_1")
            data["annotation_raw_response_2"] = row.get("annotation_raw_response_2")
            data["annotation_extracted_answer_1"] = row.get("annotation_extracted_answer_1")
            data["annotation_extracted_answer_2"] = row.get("annotation_extracted_answer_2")

            tasks.append(
                {
                    "id": i,
                    "data": data,
                    "meta": {
                        "needs_review": True,
                        "reason": self._review_reason(row),
                    },
                }
            )
        return tasks

    def _review_reason(self, row: pd.Series) -> str:
        a1 = self._normalize_answer(row.get("annotation_extracted_answer_1"))
        a2 = self._normalize_answer(row.get("annotation_extracted_answer_2"))
        if a1 is None or a2 is None:
            return "parser_failed_or_empty_answer"
        if a1 != a2:
            return "pass_disagreement"
        return "unknown"

    def _write_artifacts(
        self,
        *,
        labeled: pd.DataFrame,
        review_rows: pd.DataFrame,
        labelstudio_payload: Sequence[Mapping[str, Any]],
        quality_metrics: Mapping[str, Any],
        selected_columns: Sequence[str],
    ) -> AnnotationRunArtifacts:
        annotated_dataset_path = self.output_dir / "annotated_dataset.jsonl"
        review_path = self.output_dir / "review_rows.jsonl"
        labelstudio_path = self.output_dir / "labelstudio_review.json"
        quality_path = self.output_dir / "annotation_quality.json"
        selected_columns_path = self.output_dir / "selected_columns.json"

        labeled.to_json(annotated_dataset_path, orient="records", lines=True, force_ascii=False)
        review_rows.to_json(review_path, orient="records", lines=True, force_ascii=False)
        labelstudio_path.write_text(json.dumps(list(labelstudio_payload), indent=2, ensure_ascii=False), encoding="utf-8")

        quality_path.write_text(json.dumps(dict(quality_metrics), indent=2, ensure_ascii=False), encoding="utf-8")
        selected_columns_path.write_text(json.dumps(list(selected_columns), indent=2, ensure_ascii=False), encoding="utf-8")

        return AnnotationRunArtifacts(
            annotated_dataset_path=annotated_dataset_path,
            review_path=review_path,
            labelstudio_path=labelstudio_path,
            quality_path=quality_path,
            selected_columns_path=selected_columns_path,
        )

    def _normalize_answer(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return " ".join(text.split())

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

    def _resolve_dataframe(self, payload: Mapping[str, Any] | None) -> pd.DataFrame:
        if payload and isinstance(payload.get("dataframe"), pd.DataFrame):
            return payload["dataframe"].copy()

        input_path = self.annotation_config.get("input_path")
        if input_path:
            return self._read_dataframe(Path(input_path))

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
        configured = self.annotation_config.get("process_config", {})
        if not isinstance(configured, Mapping):
            configured = {}
        merged = {
            "parallel_workers": 200,
            "timeout_per_row": 120,
            "max_retries": 1,
            "max_tokens": 8192,
            "model": configured.get("model", "default"),
            "api_key": configured.get("api_key"),
            "headers": configured.get("headers", {}),
        }
        merged.update(dict(configured))
        return merged

    def _json_ready(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_ready(v) for v in value]
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

    def _record_log(self, message: str, logs: list[str] | None = None) -> str:
        entry = f"[{datetime.now(UTC).isoformat()}] {message}"
        if logs is not None:
            logs.append(entry)
        print(entry, file=sys.stdout, flush=True)
        return entry