import json
from string import Template
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests
import yaml

from analysis.eda import generate_eda_report
from executor.sandbox import SandboxExecutor
from models.base import BaseModelAdapter
from models.ollama_adapter import OllamaAdapter

from ..base import AgentResult, BaseAgent


UNIFIED_COLUMNS = ["text", "audio", "image", "label", "source", "collected_at", "metadata"]
SUPPORTED_SOURCE_TYPES = {"hf_dataset", "api", "scrape", "kaggle_dataset"}
SKILLS_TEMPLATE_PATH = Path(__file__).with_name("SKILLS.md")


@dataclass(slots=True)
class SourceCollectionResult:
    dataframe: pd.DataFrame
    success: bool = True
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


class DataCollectionAgent(BaseAgent):
    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        model: BaseModelAdapter | None = None,
        sandbox: SandboxExecutor | None = None,
        output_dir: str | Path = "data/raw",
    ) -> None:
        self.config = self._load_config(config)
        self.model = model or self._build_default_model()
        self.sandbox = sandbox or SandboxExecutor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sandbox_session_active = False

    def __enter__(self) -> "DataCollectionAgent":
        self.sandbox.start()
        self._sandbox_session_active = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.sandbox.close()
        self._sandbox_session_active = False

    def run(self, sources: Sequence[Mapping[str, Any]] | None = None) -> pd.DataFrame:
        payload = {"sources": list(sources)} if sources is not None else None
        result = self.execute(payload)
        if result.dataframe is None:
            raise RuntimeError("DataCollectionAgent did not produce a dataframe.")
        return result.dataframe

    def execute(self, payload: Mapping[str, Any] | None = None) -> AgentResult:
        sandbox_owned_here = False
        if not self._sandbox_session_active:
            self.sandbox.start()
            sandbox_owned_here = True

        try:
            sources = self._resolve_sources(payload)
            collected_frames: list[pd.DataFrame] = []
            logs: list[str] = []
            failed_sources: list[dict[str, Any]] = []
            source_attempts: dict[str, list[dict[str, Any]]] = {}

            for source in sources:
                source_type = source["type"]
                if source_type not in SUPPORTED_SOURCE_TYPES:
                    raise ValueError(f"Unsupported source type '{source_type}'.")

                source_result = self._collect_source(source)
                logs.extend(source_result.logs)
                source_key = self._source_key(source)
                if source_result.attempts:
                    source_attempts[source_key] = source_result.attempts
                if not source_result.success:
                    failed_sources.append(
                        {
                            "source": source_key,
                            "type": source_type,
                            "reason": source_result.logs[-1] if source_result.logs else "Collection failed.",
                        }
                    )
                    continue

                frame = source_result.dataframe
                normalized = self._normalize_frame(frame, source)
                collected_frames.append(normalized)
                logs.append(f"Collected {len(normalized)} rows from {source_key}")

            merged = self.merge(collected_frames)
            dataset_path = self.output_dir / "unified_dataset.jsonl"
            merged.to_json(dataset_path, orient="records", lines=True, force_ascii=False)

            metrics, artifacts = generate_eda_report(merged, self.output_dir / "eda")
            schema = {column: str(dtype) for column, dtype in merged.dtypes.items()}
            metrics["failed_sources"] = len(failed_sources)

            return AgentResult(
                dataframe=merged,
                dataframe_path=dataset_path,
                dataframe_schema=schema,
                metrics=metrics,
                artifacts=artifacts,
                logs=logs,
                metadata={
                    "sources": sources,
                    "failed_sources": failed_sources,
                    "source_attempts": source_attempts,
                },
            )
        finally:
            if sandbox_owned_here:
                self.sandbox.close()

    def merge(self, sources: Sequence[pd.DataFrame]) -> pd.DataFrame:
        if not sources:
            return pd.DataFrame(columns=UNIFIED_COLUMNS)
        merged = pd.concat(sources, ignore_index=True)
        return merged[UNIFIED_COLUMNS]

    def _resolve_sources(self, payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if payload and payload.get("sources"):
            return [dict(source) for source in payload["sources"]]
        return [dict(source) for source in self.config.get("sources", [])]

    def _collect_source(self, source: Mapping[str, Any]) -> SourceCollectionResult:
        source_type = source["type"]
        if source_type == "hf_dataset":
            return SourceCollectionResult(dataframe=self.load_dataset(source["name"], source="hf", options=source))
        if source_type == "kaggle_dataset":
            return SourceCollectionResult(dataframe=self.load_dataset(source["name"], source="kaggle", options=source))
        if source_type == "scrape":
            return self.scrape(source["url"], source.get("selector"), source)
        if source.get("agentic"):
            return self._run_agentic_source(source, skill=source_type)
        if source_type == "api":
            return SourceCollectionResult(
                dataframe=self.fetch_api(source["endpoint"], source.get("params"), source)
            )
        raise ValueError(f"Unsupported source type '{source_type}'.")

    def scrape(
        self,
        url: str,
        selector: str | None,
        source: Mapping[str, Any],
    ) -> SourceCollectionResult:
        source_config = dict(source)
        source_config["url"] = url
        if selector:
            source_config["selector"] = selector
        return self._run_agentic_source(source_config, skill="scrape")

    def fetch_api(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        source = source or {}
        response = requests.get(
            endpoint,
            params=params,
            headers=source.get("headers"),
            timeout=source.get("timeout", 30),
        )
        response.raise_for_status()

        payload = response.json()
        records = self._extract_records(payload, source.get("records_path"))
        if not isinstance(records, list):
            raise ValueError("API response did not resolve to a list of records.")
        return pd.DataFrame(records)

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        options: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        options = options or {}
        if source == "hf":
            from datasets import load_dataset

            dataset = load_dataset(
                name,
                options.get("config_name"),
                split=options.get("split", "train"),
            )
            if options.get("limit"):
                dataset = dataset.select(range(min(int(options["limit"]), len(dataset))))
            return dataset.to_pandas()

        if source == "kaggle":
            file_path = options.get("file_path")
            if not file_path:
                raise NotImplementedError(
                    "Kaggle support in v1 expects a local CSV or JSON file via 'file_path'."
                )
            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            if path.suffix.lower() in {".json", ".jsonl"}:
                return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
            raise ValueError(f"Unsupported Kaggle file format '{path.suffix}'.")

        raise ValueError(f"Unsupported dataset source '{source}'.")

    def _run_agentic_source(self, source: Mapping[str, Any], skill: str) -> SourceCollectionResult:
        if self.model is None:
            raise ValueError(f"{skill} sources require a model adapter.")

        schema = {
            "type": "object",
            "properties": {
                "python_code": {"type": "string"},
            },
            "required": ["python_code"],
        }
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._build_agentic_system_prompt(skill),
            },
            {
                "role": "user",
                "content": (
                    f"Generate extraction code for the '{skill}' skill using this source configuration:\n"
                    f"{json.dumps(source, indent=2)}"
                ),
            },
        ]
        attempts: list[dict[str, Any]] = []
        logs: list[str] = []
        max_attempts = max(1, int(source.get("max_attempts", 3 if skill == "scrape" else 2)))
        retry_on_empty = bool(source.get("retry_on_empty", skill == "scrape"))

        for attempt_number in range(1, max_attempts + 1):
            try:
                response = self.model.chat(messages, json_schema=schema)
                code = response["python_code"] if isinstance(response, dict) else str(response)
            except Exception as error:
                attempt_record = {
                    "attempt": attempt_number,
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "error_message": f"Model generation failed: {error}",
                    "exception_type": type(error).__name__,
                    "returncode": None,
                    "generated_code": "",
                    "row_count": 0,
                }
                attempts.append(attempt_record)
                logs.extend(self._format_attempt_logs(skill, source, attempt_record))
                if attempt_number == max_attempts:
                    break
                continue

            execution = self.sandbox.execute(code, {"source": dict(source)})

            attempt_record = {
                "attempt": attempt_number,
                "success": execution.success,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "error_message": execution.error_message,
                "exception_type": execution.exception_type,
                "returncode": execution.returncode,
                "generated_code": execution.generated_code,
                "row_count": int(len(execution.dataframe)),
            }
            attempts.append(attempt_record)
            logs.extend(self._format_attempt_logs(skill, source, attempt_record))

            if execution.success and (len(execution.dataframe) > 0 or not retry_on_empty):
                return SourceCollectionResult(
                    dataframe=execution.dataframe,
                    success=True,
                    logs=logs,
                    attempts=attempts,
                )

            if attempt_number == max_attempts:
                break

            retry_reason = execution.error_message or "Sandbox returned an empty dataframe."
            messages.extend(
                [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": self._build_retry_prompt(source, attempt_record, retry_reason),
                    },
                ]
            )

        return SourceCollectionResult(
            dataframe=pd.DataFrame(),
            success=False,
            logs=logs,
            attempts=attempts,
        )

    def _build_agentic_system_prompt(self, skill: str) -> str:
        base = self._render_skills_prompt()
        if skill == "scrape":
            return (
                f"{base} "
                "For scraping: fetch the page from context['source']['url'], parse it with BeautifulSoup, "
                "if context['source'] contains 'selector' then use it, otherwise infer a stable repeated record boundary "
                "from the page structure before extracting rows. Honor optional 'attribute' and 'limit', and return a "
                "DataFrame with the extracted fields. Prefer a 'text' column when only one value is available."
            )
        if skill == "api":
            return (
                f"{base} "
                "For APIs: call context['source']['endpoint'] with optional params and headers, convert the JSON payload "
                "into a list of records, and return a DataFrame."
            )
        return base

    def _render_skills_prompt(self) -> str:
        requirements = "\n".join(
            f"- `{module}`" for module in sorted(self.sandbox.allowed_imports)
        )
        template = Template(SKILLS_TEMPLATE_PATH.read_text(encoding="utf-8"))
        return template.substitute(requirements=requirements)

    def _build_retry_prompt(
        self,
        source: Mapping[str, Any],
        attempt_record: Mapping[str, Any],
        retry_reason: str,
    ) -> str:
        return (
            "The previous generated code did not produce an acceptable result. "
            "Revise the code and return only JSON with a new 'python_code' value.\n"
            f"Source configuration:\n{json.dumps(dict(source), indent=2)}\n"
            f"Failure reason: {retry_reason}\n"
            f"stdout:\n{attempt_record.get('stdout') or '<empty>'}\n"
            f"stderr:\n{attempt_record.get('stderr') or '<empty>'}\n"
            f"Previous row count: {attempt_record.get('row_count')}"
        )

    def _format_attempt_logs(
        self,
        skill: str,
        source: Mapping[str, Any],
        attempt_record: Mapping[str, Any],
    ) -> list[str]:
        source_key = self._source_key(source)
        status = "succeeded" if attempt_record["success"] else "failed"
        lines = [
            f"{skill} attempt {attempt_record['attempt']} for {source_key} {status} with {attempt_record['row_count']} rows.",
        ]
        if attempt_record.get("error_message"):
            lines.append(
                f"{skill} attempt {attempt_record['attempt']} error for {source_key}: "
                f"{attempt_record['error_message']}"
            )
        if attempt_record.get("stderr"):
            lines.append(
                f"{skill} attempt {attempt_record['attempt']} stderr for {source_key}: "
                f"{attempt_record['stderr'].strip()}"
            )
        return lines

    def _normalize_frame(self, frame: pd.DataFrame, source: Mapping[str, Any]) -> pd.DataFrame:
        working = frame.copy()
        working.columns = [str(column) for column in working.columns]

        column_map = source.get("column_map", {})
        reverse_map = {raw_name: unified_name for unified_name, raw_name in column_map.items()}
        if reverse_map:
            working = working.rename(columns=reverse_map)

        for unified_name in ["text", "audio", "image", "label"]:
            if unified_name not in working.columns:
                inferred = self._infer_column(working, unified_name)
                if inferred:
                    working[unified_name] = working[inferred]
                else:
                    working[unified_name] = None

        metadata_columns = [column for column in working.columns if column not in {"text", "audio", "image", "label"}]
        working["metadata"] = working[metadata_columns].to_dict(orient="records")
        working["source"] = source.get("name") or source.get("url") or source.get("endpoint") or source["type"]
        working["collected_at"] = datetime.now(UTC).isoformat()
        return working[UNIFIED_COLUMNS]

    def _infer_column(self, frame: pd.DataFrame, unified_name: str) -> str | None:
        candidate_map = {
            "text": ["text", "content", "body", "review", "comment", "description"],
            "audio": ["audio", "audio_path", "file", "path"],
            "image": ["image", "image_path", "url"],
            "label": ["label", "labels", "sentiment", "target", "class"],
        }
        for candidate in candidate_map[unified_name]:
            if candidate in frame.columns:
                return candidate
        return None

    def _extract_records(self, payload: Any, records_path: str | None) -> Any:
        if records_path is None:
            return payload

        current = payload
        for key in records_path.split("."):
            if isinstance(current, list):
                current = current[int(key)]
            else:
                current = current[key]
        return current

    def _source_key(self, source: Mapping[str, Any]) -> str:
        return str(source.get("name") or source.get("url") or source.get("endpoint") or source["type"])

    def _load_config(self, config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(config, Mapping):
            return dict(config)

        path = Path(config)
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _build_default_model(self) -> BaseModelAdapter | None:
        llm_config = dict(self.config.get("llm", {}))
        if llm_config and llm_config.get("provider", "ollama") != "ollama":
            return None
        if not llm_config and not self._requires_agentic_model():
            return None

        return OllamaAdapter(
            model=llm_config.get("model", "kimi-k2.5:cloud"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
        )

    def _requires_agentic_model(self) -> bool:
        for source in self.config.get("sources", []):
            if source.get("type") == "scrape" or source.get("agentic"):
                return True
        return False
