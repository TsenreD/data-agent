import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from agents.data_collection.smolagents_backend import (
    EDA_NOTEBOOK_IMPORTS,
    _SmolagentsLocalBackendBase,
    _parse_json_payload,
)

from .skillset import load_skill


@dataclass(slots=True)
class AnnotationSetupResult:
    columns: list[str]
    examples: list[dict[str, Any]]
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParserResult:
    answers: list[Any]
    normalized_outputs: list[Any] = field(default_factory=list)
    success: bool = False
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class _AnnotationEdaBackend(_SmolagentsLocalBackendBase):
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config, agent_config=agent_config, model=model)

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
    def _format_attempt_logs(prefix: str, attempt: Mapping[str, Any]) -> list[str]:
        status = "succeeded" if attempt["success"] else "failed"
        lines = [f"{prefix} attempt {attempt['attempt']} {status}."]
        if attempt.get("error_message"):
            lines.append(f"{prefix} attempt {attempt['attempt']} error: {attempt['error_message']}")
        notes = attempt.get("notes") or []
        if notes:
            lines.append(f"{prefix} attempt {attempt['attempt']} notes: " + " | ".join(str(note) for note in notes))
        return lines

class SmolagentsAnnotationBackend(_AnnotationEdaBackend):
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config, agent_config=agent_config, model=model)
        self.instructions = load_skill("auto_label")

    def select_columns_with_fewshot(
        self,
        *,
        user_prompt: str,
        columns: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        task = self._build_task(user_prompt=user_prompt, columns=columns)
        result = self._run_json_skill(
            task=task,
            instructions=self.instructions,
            parser=self._parse_output,
            result_type=AnnotationSetupResult,
            failure_kwargs={"columns": [], "examples": []},
            log_prefix="annotation setup",
            source={},
            max_steps=int(self.agent_config.get("max_steps", 6)),
            max_attempts=int(self.agent_config.get("max_attempts", 2)),
        )
        if not result.success:
            raise RuntimeError(result.notes[-1] if result.notes else "annotation setup failed")
        return result.columns, result.examples

    def _build_task(
        self,
        *,
        user_prompt: str,
        columns: list[str],
    ) -> str:
        return (
            "Prepare a compact LLM annotation setup.\n"
            f"User prompt: {user_prompt or '<none provided>'}\n"
            f"Selected input columns: {json.dumps(columns, ensure_ascii=False)}\n"
            "Do not inspect any dataset rows.\n"
            "Keep the provided columns unless they are obviously invalid.\n"
            "Synthesize a small, high-signal few-shot set from the prompt and column names only.\n"
            "Return strict JSON with top-level fields `columns` and `examples`.\n"
            "Keep `columns` minimal and `examples` high-signal."
        )

    @staticmethod
    def _parse_output(raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict):
            raise ValueError("Annotation setup output was not a JSON object.")
        columns = payload.get("columns", [])
        if not isinstance(columns, list):
            columns = []
        examples = payload.get("examples", [])
        if not isinstance(examples, list):
            examples = []
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return {
            "columns": [str(item) for item in columns],
            "examples": [dict(item) for item in examples if isinstance(item, Mapping)],
        }, [str(note) for note in notes]


class SmolagentsParserBackend(_AnnotationEdaBackend):
    BOXED_PATTERN = re.compile(r"\\boxed\s*{([^{}]+)}")
    JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"(?im)^\s*(?:final\s+answer|answer)\s*[:=]\s*(.+?)\s*$")
    INVALID_PATTERNS = (
        "incomplete",
        "insufficient information",
        "cannot be determined",
        "can't be determined",
        "cannot determine",
        "not enough information",
        "no solution",
        "unsolved",
        "unanswerable",
        "invalid",
    )

    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(llm_config=llm_config, agent_config=agent_config, model=model)
        self.instructions = ""

    def parse_answers(
        self,
        df_path: str | Path,
        *,
        column_name: str,
        task_prompt: str,
        selected_columns: Sequence[str] | None = None,
        sample_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        dataset_path = Path(df_path)
        dataframe = self._read_dataframe(dataset_path)
        if column_name not in dataframe.columns:
            raise ValueError(f"Column `{column_name}` was not found in parser input dataset.")

        answers: list[Any] = []
        normalized_outputs: list[Any] = []
        notes: list[str] = []
        for raw_value in dataframe[column_name].tolist():
            parsed = self._parse_single_output(raw_value)
            answers.append(parsed)
            normalized_outputs.append(self._normalize_output(raw_value))

        result = ParserResult(
            answers=answers,
            normalized_outputs=normalized_outputs,
            success=True,
            logs=[f"annotation parser parsed {len(answers)} rows deterministically."],
            attempts=[],
            notes=notes,
        )
        return {
            "answers": result.answers,
            "normalized_outputs": result.normalized_outputs,
            "notes": result.notes,
        }

    @staticmethod
    def _read_dataframe(dataset_path: Path) -> pd.DataFrame:
        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        if suffix == ".json":
            return pd.read_json(dataset_path)
        if suffix == ".jsonl":
            return pd.read_json(dataset_path, lines=True)
        raise ValueError(f"Unsupported parser dataset format: {dataset_path.suffix}")

    def _parse_single_output(self, raw_value: Any) -> str | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, Mapping):
            return self._extract_from_mapping(raw_value)
        if isinstance(raw_value, list):
            return self._normalize_candidate(raw_value[-1]) if raw_value else None

        text = str(raw_value).strip()
        if not text:
            return None

        payload = self._extract_json_payload(text)
        if isinstance(payload, Mapping):
            parsed = self._extract_from_mapping(payload)
            if parsed is not None:
                return parsed

        boxed_match = self.BOXED_PATTERN.search(text)
        if boxed_match:
            return self._normalize_candidate(boxed_match.group(1))

        lowered = text.lower()
        if any(pattern in lowered for pattern in self.INVALID_PATTERNS):
            return "invalid"

        answer_matches = self.ANSWER_PATTERN.findall(text)
        if answer_matches:
            return self._normalize_candidate(answer_matches[-1])

        return None

    def _extract_from_mapping(self, payload: Mapping[str, Any]) -> str | None:
        for key in ("label", "final_label", "answer"):
            if key in payload:
                value = self._normalize_candidate(payload.get(key))
                if value:
                    return value
        return None

    def _extract_json_payload(self, text: str) -> Any | None:
        fenced_match = self.JSON_FENCE_PATTERN.search(text)
        if fenced_match:
            text = fenced_match.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _normalize_output(raw_value: Any) -> str | None:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        return text or None

    def _normalize_candidate(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if re.fullmatch(r"[.]+|[…]+", text):
            return None
        boxed_match = self.BOXED_PATTERN.search(text)
        if boxed_match:
            text = boxed_match.group(1).strip()
        if text.lower().startswith("answer:"):
            text = text.split(":", 1)[1].strip()
        if not text:
            return None
        return text
