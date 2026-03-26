import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from agents.data_collection.smolagents_backend import (
    EDA_NOTEBOOK_IMPORTS,
    _SmolagentsLocalBackendBase,
    _parse_json_payload,
)

from .skillset import load_skill


@dataclass(slots=True)
class QualityDetectionResult:
    report: dict[str, Any] | None
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityFixResult:
    summary: dict[str, Any] | None
    strategy_used: dict[str, Any] = field(default_factory=dict)
    justification: str = ""
    success: bool = False
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityComparisonResult:
    comparison: dict[str, Any] | None
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityAnalysisResult:
    analysis: dict[str, Any] | None
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SmolagentsQualityBackend(_SmolagentsLocalBackendBase):
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config, agent_config=agent_config, model=model)
        self.analyze_instructions = load_skill("analyze")
        self.detect_instructions = load_skill("detect_issues")
        self.fix_instructions = load_skill("fix")
        self.compare_instructions = load_skill("compare")

    def detect_issues(
        self,
        dataset_path: str | Path,
        *,
        project_context: Mapping[str, Any] | None = None,
        label_column: str | None = None,
        imbalance_threshold: float = 0.75,
        preview_frame: Any | None = None,
    ) -> QualityDetectionResult:
        dataset_path = Path(dataset_path)
        task = self._build_detect_task(
            dataset_path,
            project_context=project_context,
            label_column=label_column,
            imbalance_threshold=imbalance_threshold,
            preview_frame=preview_frame,
        )
        return self._run_json_skill(
            task=task,
            instructions=self.detect_instructions,
            parser=self._parse_detect_output,
            result_type=QualityDetectionResult,
            failure_kwargs={"report": None},
            log_prefix="quality detection",
            source={},
            max_steps=int(self.agent_config.get("detect_max_steps", self.agent_config.get("max_steps", 10))),
            max_attempts=int(self.agent_config.get("detect_max_attempts", self.agent_config.get("max_attempts", 2))),
        )

    def analyze(
        self,
        dataset_path: str | Path,
        *,
        detection_report: Mapping[str, Any],
        project_context: Mapping[str, Any] | None = None,
        task_description: str = "",
        strategy_hint: Mapping[str, Any] | None = None,
        label_column: str | None = None,
        preview_frame: Any | None = None,
    ) -> QualityAnalysisResult:
        dataset_path = Path(dataset_path)
        task = self._build_analyze_task(
            dataset_path,
            detection_report=detection_report,
            project_context=project_context,
            task_description=task_description,
            strategy_hint=strategy_hint,
            label_column=label_column,
            preview_frame=preview_frame,
        )
        return self._run_json_skill(
            task=task,
            instructions=self.analyze_instructions,
            parser=self._parse_analyze_output,
            result_type=QualityAnalysisResult,
            failure_kwargs={"analysis": None},
            log_prefix="quality analysis",
            source={},
            max_steps=int(self.agent_config.get("analyze_max_steps", self.agent_config.get("max_steps", 8))),
            max_attempts=int(self.agent_config.get("analyze_max_attempts", self.agent_config.get("max_attempts", 2))),
        )

    def fix(
        self,
        dataset_path: str | Path,
        output_path: str | Path,
        *,
        strategy: Mapping[str, Any],
        task_description: str = "",
        analysis_context: Mapping[str, Any] | None = None,
        label_column: str | None = None,
        imbalance_threshold: float = 0.75,
    ) -> QualityFixResult:
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        task = self._build_fix_task(
            dataset_path,
            output_path,
            strategy=strategy,
            task_description=task_description,
            analysis_context=analysis_context,
            label_column=label_column,
            imbalance_threshold=imbalance_threshold,
        )
        return self._run_json_skill(
            task=task,
            instructions=self.fix_instructions,
            parser=self._parse_fix_output,
            result_type=QualityFixResult,
            failure_kwargs={"summary": None, "strategy_used": dict(strategy), "justification": ""},
            log_prefix="quality fix",
            source={},
            max_steps=int(self.agent_config.get("fix_max_steps", self.agent_config.get("max_steps", 12))),
            max_attempts=int(self.agent_config.get("fix_max_attempts", self.agent_config.get("max_attempts", 2))),
        )

    def compare(
        self,
        before_path: str | Path,
        after_path: str | Path,
        *,
        analysis_context: Mapping[str, Any] | None = None,
        label_column: str | None = None,
        imbalance_threshold: float = 0.75,
    ) -> QualityComparisonResult:
        before_path = Path(before_path)
        after_path = Path(after_path)
        task = self._build_compare_task(
            before_path,
            after_path,
            analysis_context=analysis_context,
            label_column=label_column,
            imbalance_threshold=imbalance_threshold,
        )
        return self._run_json_skill(
            task=task,
            instructions=self.compare_instructions,
            parser=self._parse_compare_output,
            result_type=QualityComparisonResult,
            failure_kwargs={"comparison": None},
            log_prefix="quality comparison",
            source={},
            max_steps=int(self.agent_config.get("compare_max_steps", self.agent_config.get("max_steps", 10))),
            max_attempts=int(self.agent_config.get("compare_max_attempts", self.agent_config.get("max_attempts", 2))),
        )

    def _run_json_skill(
        self,
        *,
        task: str,
        instructions: str,
        parser: Callable[[Any], tuple[dict[str, Any], list[str]]],
        result_type: type[Any],
        failure_kwargs: dict[str, Any],
        log_prefix: str,
        source: Mapping[str, Any],
        max_steps: int,
        max_attempts: int,
    ) -> Any:
        logs: list[str] = []
        attempts: list[dict[str, Any]] = []

        for attempt_number in range(1, max(1, max_attempts) + 1):
            attempt_task = self._build_attempt_task(task, attempts[-1] if attempts else None)
            try:
                run_result = self._run_agent(
                    task=attempt_task,
                    instructions=instructions,
                    tools=[],
                    additional_imports=EDA_NOTEBOOK_IMPORTS,
                    max_steps=max(1, max_steps),
                    source=source,
                )
                raw_output = "" if run_result.output is None else str(run_result.output)
                parsed_payload, notes = parser(raw_output)
                attempt = {
                    "attempt": attempt_number,
                    "success": True,
                    "output": raw_output,
                    "state": run_result.state,
                    "notes": notes,
                    "steps": run_result.steps or [],
                }
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(log_prefix, attempt))
                return result_type(**parsed_payload, success=True, logs=logs, attempts=attempts, notes=notes)
            except Exception as error:
                attempt = {
                    "attempt": attempt_number,
                    "success": False,
                    "output": "",
                    "state": "error",
                    "notes": [str(error)],
                    "steps": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(log_prefix, attempt))

        error_notes = [str(attempts[-1].get("error_message", ""))] if attempts else []
        return result_type(**failure_kwargs, success=False, logs=logs, attempts=attempts, notes=error_notes)

    def _build_detect_task(
        self,
        dataset_path: Path,
        *,
        project_context: Mapping[str, Any] | None,
        label_column: str | None,
        imbalance_threshold: float,
        preview_frame: Any | None,
    ) -> str:
        local_dataset_path = str(dataset_path.resolve())
        project_json = json.dumps(dict(project_context or {}), ensure_ascii=False)
        preview_json, schema_json = self._preview_and_schema_json(preview_frame)
        primary_modality = self._primary_modality(project_context, preview_frame)
        return (
            "Detect data quality issues in the real dataset by writing Python code and inspecting the actual rows.\n"
            f"Host dataset path: {dataset_path.as_posix()}\n"
            f"Local dataset path: {local_dataset_path}\n"
            f"Project context: {project_json}\n"
            f"Primary modality: {primary_modality}\n"
            f"Preferred label column: {label_column or '<infer>'}\n"
            f"Imbalance threshold: {imbalance_threshold}\n"
            f"Schema: {schema_json}\n"
            f"Preview rows: {preview_json}\n"
            "Load the dataset with pandas using the local path.\n"
            "Return strict JSON with top-level fields `report` and optional `notes`.\n"
            "Write one end-to-end code block that computes the report and calls `final_answer(...)` directly.\n"
            "Do not spend steps on exploratory prints unless code execution fails.\n"
            "If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.\n"
            "This is the Detective stage from the spec: include the required generic metrics, but make the main "
            "modality the center of the report.\n"
            "If the primary modality is text, inspect text length distribution, empty or near-empty prompts, obvious "
            "truncation, exact or normalized duplicate prompts, near-duplicate prompts, source-to-label coverage, and optional modality "
            "columns that are entirely null.\n"
            "Treat deduplication as a core quality decision: quantify exact duplicate rows and text-level duplicates, "
            "including duplicates that differ only by whitespace, punctuation, or metadata noise.\n"
            "Measure row sparsity as well: identify rows where most fields are null or the primary modality content is "
            "missing, and separate those from informative rows that merely lack labels.\n"
            "If the preferred label column is high-cardinality numeric data or behaves like a regression target, "
            "set imbalance as not applicable instead of forcing a class-balance analysis.\n"
            "Compute missing values, duplicate rows, numeric outliers, and class imbalance from the real data.\n"
            "Also include modality-aware findings such as `modality_profile`, `text_profile`, `task_signals`, or "
            "`sparsity_profile` when they help explain what should actually be cleaned."
        )

    def _build_analyze_task(
        self,
        dataset_path: Path,
        *,
        detection_report: Mapping[str, Any],
        project_context: Mapping[str, Any] | None,
        task_description: str,
        strategy_hint: Mapping[str, Any] | None,
        label_column: str | None,
        preview_frame: Any | None,
    ) -> str:
        local_dataset_path = str(dataset_path.resolve())
        report_json = json.dumps(dict(detection_report), ensure_ascii=False)
        project_json = json.dumps(dict(project_context or {}), ensure_ascii=False)
        strategy_json = json.dumps(dict(strategy_hint or {}), ensure_ascii=False)
        preview_json, schema_json = self._preview_and_schema_json(preview_frame)
        primary_modality = self._primary_modality(project_context, preview_frame)
        return (
            "Interpret the dataset and recommend a meaningful quality strategy for the actual ML task.\n"
            f"Host dataset path: {dataset_path.as_posix()}\n"
            f"Local dataset path: {local_dataset_path}\n"
            f"Project context: {project_json}\n"
            f"Primary modality: {primary_modality}\n"
            f"Task description: {task_description or '<none provided>'}\n"
            f"Preferred label column: {label_column or '<infer>'}\n"
            f"Schema: {schema_json}\n"
            f"Preview rows: {preview_json}\n"
            f"Detected issue report: {report_json}\n"
            f"Current strategy hint: {strategy_json}\n"
            "Load the dataset with pandas and inspect actual rows before deciding.\n"
            "Return strict JSON with top-level fields `analysis` and optional `notes`.\n"
            "This is the Analyzer stage from the spec.\n"
            "Your `analysis` object must include `task_interpretation`, `recommended_strategy`, "
            "`alternative_strategies`, `quality_focus`, and `justification`.\n"
            "Recommend at least two plausible cleaning strategies, then identify the best one for the likely ML task.\n"
            "Do not spend a step on print-only debugging or exploratory display calls.\n"
            "Use the preview/schema context plus a small amount of real inspection, then return the final JSON directly.\n"
            "Explain which checks are meaningful, which are misleading, and what cleaning strategy best fits this dataset.\n"
            "Let modality drive the plan. For text datasets, prioritize prompt quality, text length and emptiness, "
            "duplicate or templated problem statements, label coverage by source, answer-format semantics, and whether "
            "non-text columns should be dropped as irrelevant empty modalities.\n"
            "Make deduplication policy explicit: say whether to drop exact duplicates, normalized text duplicates, or "
            "near-duplicates, and explain when superficially similar rows should still be kept.\n"
            "Unless the task description explicitly asks for a supervised-only subset, optimize for maximum useful data "
            "retention in the cleaned dataset.\n"
            "Your recommended strategy may include modality-specific actions such as `text_actions`, `row_actions`, "
            "`column_actions`, `target_actions`, or `label_actions`; do not limit yourself to generic imputation rules.\n"
            "Missing labels are not an automatic reason to drop rows. Distinguish between low-information rows that "
            "should be removed and informative unlabeled rows that should be preserved, flagged, or kept for later "
            "label generation.\n"
            "If you want a supervised-only subset, present it as one strategy option, but prefer a retention-oriented "
            "cleaned dataset unless the task description clearly demands otherwise.\n"
            "Rows with substantial text or metadata should usually be kept even if `label` is null."
        )

    def _build_fix_task(
        self,
        dataset_path: Path,
        output_path: Path,
        *,
        strategy: Mapping[str, Any],
        task_description: str,
        analysis_context: Mapping[str, Any] | None,
        label_column: str | None,
        imbalance_threshold: float,
    ) -> str:
        local_dataset_path = str(dataset_path.resolve())
        local_output_path = str(output_path.resolve())
        strategy_json = json.dumps(dict(strategy), ensure_ascii=False)
        analysis_json = json.dumps(dict(analysis_context or {}), ensure_ascii=False)
        primary_modality = self._primary_modality(None, None, analysis_context)
        return (
            "Clean the real dataset by writing Python code, applying the requested strategy, and saving the cleaned result.\n"
            f"Host input dataset path: {dataset_path.as_posix()}\n"
            f"Local input dataset path: {local_dataset_path}\n"
            f"Host output dataset path: {output_path.as_posix()}\n"
            f"Local output dataset path: {local_output_path}\n"
            f"Primary modality: {primary_modality}\n"
            f"Requested strategy: {strategy_json}\n"
            f"Task description: {task_description or 'general ML-ready tabular/text cleaning'}\n"
            f"Task-aware analysis context: {analysis_json}\n"
            f"Preferred label column: {label_column or '<infer>'}\n"
            f"Imbalance threshold: {imbalance_threshold}\n"
            "Load the input dataset with pandas, apply the strategy directly in code, and write JSONL to the local output path.\n"
            "Write one end-to-end code block that cleans, writes, and calls `final_answer(...)` directly.\n"
            "Do not spend steps on exploratory prints unless code execution fails.\n"
            "If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.\n"
            "Use the task-aware analysis context to decide which columns to preserve, which issues to ignore, and whether target cleaning is appropriate.\n"
            "If the strategy includes modality-specific actions such as `text_actions`, `row_actions`, `column_actions`, "
            "`target_actions`, or `label_actions`, execute them directly instead of collapsing them into generic tabular rules.\n"
            "Perform deduplication early when duplicates are present. For text datasets, deduplicate on normalized prompt "
            "content when appropriate, not only exact whole-row equality.\n"
            "For text datasets, prefer actions like dropping empty prompts, removing exact duplicate prompts, trimming "
            "whitespace, preserving LaTeX or math syntax, dropping all-null non-text modality columns, and making an "
            "explicit decision about unlabeled rows based on the inferred task.\n"
            "If duplicate rows differ only in metadata richness or label availability, prefer keeping the richer row when "
            "that choice is obvious from the data.\n"
            "The cleaned dataset should maximize retention of useful data unless the task description explicitly requires "
            "a supervised-only subset.\n"
            "Do not drop rows only because `label` is null unless the strategy explicitly requires it. Prefer dropping "
            "rows that are mostly empty, missing the primary modality content, or otherwise low-information.\n"
            "If the strategy says unlabeled rows may be useful later, keep them and mark or summarize that choice.\n"
            "Return strict JSON with top-level fields `summary`, `strategy_used`, `justification`, and optional `notes`.\n"
            "Do not merely describe the cleaning steps; execute them and persist the output dataset."
        )

    def _build_compare_task(
        self,
        before_path: Path,
        after_path: Path,
        *,
        analysis_context: Mapping[str, Any] | None,
        label_column: str | None,
        imbalance_threshold: float,
    ) -> str:
        local_before_path = str(before_path.resolve())
        local_after_path = str(after_path.resolve())
        analysis_json = json.dumps(dict(analysis_context or {}), ensure_ascii=False)
        primary_modality = self._primary_modality(None, None, analysis_context)
        return (
            "Compare the real before and after datasets by writing Python code and computing quality metrics for both.\n"
            f"Host before dataset path: {before_path.as_posix()}\n"
            f"Local before dataset path: {local_before_path}\n"
            f"Host after dataset path: {after_path.as_posix()}\n"
            f"Local after dataset path: {local_after_path}\n"
            f"Primary modality: {primary_modality}\n"
            f"Task-aware analysis context: {analysis_json}\n"
            f"Preferred label column: {label_column or '<infer>'}\n"
            f"Imbalance threshold: {imbalance_threshold}\n"
            "Load both datasets with pandas from the local paths.\n"
            "Write one end-to-end code block that computes the comparison and calls `final_answer(...)` directly.\n"
            "If object columns contain dicts or lists, normalize them to stable JSON strings before duplicate checks.\n"
            "Use the task-aware analysis context to emphasize meaningful metrics and mark irrelevant checks as not applicable.\n"
            "If the primary modality is text, also compare text-centric effects such as empty prompt count, exact "
            "duplicate prompt count, text length summaries, label coverage, and removal of all-null auxiliary modality columns.\n"
            "Show duplicate reduction clearly: include exact duplicate rows and normalized duplicate text before and after.\n"
            "If retention versus dropping unlabeled rows was part of the decision, compare how many informative unlabeled "
            "rows were preserved versus how many sparse rows were removed.\n"
            "Return strict JSON with top-level fields `comparison` and optional `notes`.\n"
            "The comparison must include before/after values for missing values, duplicates, outliers, imbalance, and row count."
        )

    def _build_attempt_task(
        self,
        base_task: str,
        previous_attempt: Mapping[str, Any] | None,
    ) -> str:
        if previous_attempt is None:
            return base_task
        return (
            f"{base_task}\n\n"
            "Previous output was rejected.\n"
            f"Previous error: {previous_attempt.get('error_message') or 'unknown error'}\n"
            f"Previous notes: {json.dumps(previous_attempt.get('notes', []), ensure_ascii=False)}\n"
            f"Previous output:\n{previous_attempt.get('output') or '<empty>'}\n"
            "Revise the generated code materially and return strict JSON only."
        )

    @staticmethod
    def _parse_detect_output(raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict):
            raise ValueError("Quality detection output did not include a report object.")
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return {"report": payload["report"]}, [str(note) for note in notes]

    @staticmethod
    def _parse_analyze_output(raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict) or not isinstance(payload.get("analysis"), dict):
            raise ValueError("Quality analysis output did not include an analysis object.")
        analysis = dict(payload["analysis"])
        task_interpretation = analysis.get("task_interpretation")
        if isinstance(task_interpretation, Mapping):
            analysis["task_interpretation"] = dict(task_interpretation)
        elif task_interpretation:
            analysis["task_interpretation"] = {"summary": str(task_interpretation)}
        else:
            analysis["task_interpretation"] = {}
        recommended_strategy = analysis.get("recommended_strategy")
        analysis["recommended_strategy"] = (
            dict(recommended_strategy) if isinstance(recommended_strategy, Mapping) else {}
        )
        alternative_strategies = analysis.get("alternative_strategies")
        analysis["alternative_strategies"] = (
            list(alternative_strategies) if isinstance(alternative_strategies, list) else []
        )
        quality_focus = analysis.get("quality_focus")
        analysis["quality_focus"] = dict(quality_focus) if isinstance(quality_focus, Mapping) else {}
        analysis["justification"] = str(analysis.get("justification", "")).strip()
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return {"analysis": analysis}, [str(note) for note in notes]

    @staticmethod
    def _parse_fix_output(raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
            raise ValueError("Quality fix output did not include a summary object.")
        strategy_used = payload.get("strategy_used", {})
        if not isinstance(strategy_used, dict):
            strategy_used = {}
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return {
            "summary": payload["summary"],
            "strategy_used": strategy_used,
            "justification": str(payload.get("justification", "")).strip(),
        }, [str(note) for note in notes]

    @staticmethod
    def _parse_compare_output(raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict) or not isinstance(payload.get("comparison"), dict):
            raise ValueError("Quality comparison output did not include a comparison object.")
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return {"comparison": payload["comparison"]}, [str(note) for note in notes]

    @staticmethod
    def _format_attempt_logs(prefix: str, attempt: Mapping[str, Any]) -> list[str]:
        status = "succeeded" if attempt["success"] else "failed"
        lines = [f"{prefix} attempt {attempt['attempt']} {status}."]
        if attempt.get("error_message"):
            lines.append(f"{prefix} attempt {attempt['attempt']} error: {attempt['error_message']}")
        notes = attempt.get("notes") or []
        if notes:
            lines.append(f"{prefix} attempt {attempt['attempt']} notes: " + " | ".join(str(note) for note in notes))
        return lines

    @staticmethod
    def _preview_and_schema_json(preview_frame: Any | None) -> tuple[str, str]:
        if preview_frame is None:
            return "[]", "{}"
        try:
            preview_json = preview_frame.head(min(3, len(preview_frame))).to_json(
                orient="records",
                force_ascii=False,
                date_format="iso",
            )
            schema_json = json.dumps(
                {column: str(dtype) for column, dtype in preview_frame.dtypes.items()},
                ensure_ascii=False,
            )
            return preview_json, schema_json
        except Exception:
            return "[]", "{}"

    @staticmethod
    def _primary_modality(
        project_context: Mapping[str, Any] | None,
        preview_frame: Any | None,
        analysis_context: Mapping[str, Any] | None = None,
    ) -> str:
        task_interpretation = (
            dict((analysis_context or {}).get("task_interpretation", {}))
            if isinstance((analysis_context or {}).get("task_interpretation"), Mapping)
            else {}
        )
        quality_focus = (
            dict((analysis_context or {}).get("quality_focus", {}))
            if isinstance((analysis_context or {}).get("quality_focus"), Mapping)
            else {}
        )
        candidates = [
            task_interpretation.get("primary_modality"),
            quality_focus.get("primary_modality"),
            (project_context or {}).get("modality"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if preview_frame is not None:
            columns = {str(column).lower() for column in getattr(preview_frame, "columns", [])}
            if "text" in columns:
                return "text"
            if "image" in columns:
                return "image"
            if "audio" in columns:
                return "audio"
        return "unknown"
