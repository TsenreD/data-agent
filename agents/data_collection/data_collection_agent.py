import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import requests
import yaml

from ..base import AgentResult, BaseAgent
from .smolagents_backend import SmolagentsCollectionBackend, SmolagentsNotebookBackend


UNIFIED_COLUMNS = ["text", "audio", "image", "label", "source", "collected_at", "metadata"]
SUPPORTED_SOURCE_TYPES = {"hf_dataset", "api", "scrape", "kaggle_dataset"}


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
        output_dir: str | Path = "data/raw",
        notebook_path: str | Path = "notebooks/eda.ipynb",
    ) -> None:
        self.config = self._load_config(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.notebook_path = Path(notebook_path)
        self.collection_backend = SmolagentsCollectionBackend(
            llm_config=self.config.get("llm", {}),
        )
        self.notebook_backend = SmolagentsNotebookBackend(
            llm_config=self.config.get("llm", {}),
        )

    def run(self, sources: Sequence[Mapping[str, Any]] | None = None) -> pd.DataFrame:
        payload = {"sources": list(sources)} if sources is not None else None
        result = self.execute(payload)
        if result.dataframe is None:
            raise RuntimeError("DataCollectionAgent did not produce a dataframe.")
        return result.dataframe

    def execute(self, payload: Mapping[str, Any] | None = None) -> AgentResult:
        logs: list[str] = []
        self._record_log("Starting DataCollectionAgent execution.", logs)

        try:
            sources = self._resolve_sources(payload)
            collected_frames: list[pd.DataFrame] = []
            failed_sources: list[dict[str, Any]] = []
            source_attempts: dict[str, list[dict[str, Any]]] = {}

            for source in sources:
                source_type = source["type"]
                if source_type not in SUPPORTED_SOURCE_TYPES:
                    raise ValueError(f"Unsupported source type '{source_type}'.")

                source_key = self._source_key(source)
                self._record_log(f"Starting source {source_key} ({source_type}).", logs)
                source_result = self._collect_source(source)
                self._record_existing_logs(source_result.logs, logs)
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
                    self._record_log(f"Source {source_key} failed; continuing with remaining sources.", logs)
                    continue

                normalized = self._normalize_frame(source_result.dataframe, source)
                collected_frames.append(normalized)
                self._record_log(f"Collected {len(normalized)} rows from {source_key}.", logs)

            merged = self.merge(collected_frames)
            dataset_path = self.output_dir / "unified_dataset.jsonl"
            merged.to_json(dataset_path, orient="records", lines=True, force_ascii=False)
            self._record_log(f"Wrote merged dataset to {dataset_path}.", logs)

            notebook_result = self.notebook_backend.generate_notebook(
                frame=merged,
                dataset_path=dataset_path,
                notebook_path=self.notebook_path,
            )
            self._record_existing_logs(notebook_result.logs, logs)
            if not notebook_result.success or notebook_result.notebook is None:
                notebook_error = self._summarize_notebook_failure(notebook_result)
                raise RuntimeError(f"EDA notebook generation failed: {notebook_error}")
            self.notebook_path.parent.mkdir(parents=True, exist_ok=True)
            self.notebook_path.write_text(
                json.dumps(notebook_result.notebook, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            schema = {column: str(dtype) for column, dtype in merged.dtypes.items()}
            metrics = {
                "row_count": int(len(merged)),
                "failed_sources": len(failed_sources),
            }
            artifacts = {"eda_notebook": str(self.notebook_path)}
            self._record_log(
                f"Generated EDA notebook at {self.notebook_path}.",
                logs,
            )
            self._record_log("DataCollectionAgent execution finished successfully.", logs)

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
                    "eda_notebook_notes": notebook_result.notes,
                },
            )
        except Exception as error:
            self._record_log(
                f"DataCollectionAgent execution failed with {type(error).__name__}: {error}",
                logs,
            )
            raise

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
            return SourceCollectionResult(
                dataframe=self.load_dataset(source["name"], source="kaggle", options=source)
            )
        if source_type == "scrape":
            return self.scrape(source["url"], source.get("selector"), source)
        if source_type == "api" and source.get("agentic"):
            return self._run_agentic_source(source)
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
        return self._run_agentic_source(source_config)

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
                    "Kaggle support expects a local CSV or JSON file via 'file_path'."
                )
            path = Path(file_path)
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            if path.suffix.lower() in {".json", ".jsonl"}:
                return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
            raise ValueError(f"Unsupported Kaggle file format '{path.suffix}'.")

        raise ValueError(f"Unsupported dataset source '{source}'.")

    def _run_agentic_source(self, source: Mapping[str, Any]) -> SourceCollectionResult:
        result = self.collection_backend.collect(source)
        frame = pd.DataFrame(result.records)
        return SourceCollectionResult(
            dataframe=frame,
            success=result.success,
            logs=result.logs,
            attempts=result.attempts,
        )

    def _record_existing_logs(self, entries: Sequence[str], logs: list[str]) -> None:
        for entry in entries:
            if entry in logs:
                continue
            logs.append(entry)

    def _record_log(
        self,
        message: str,
        logs: list[str] | None = None,
        already_formatted: bool = False,
    ) -> str:
        entry = message if already_formatted else f"[{datetime.now(UTC).isoformat()}] {message}"
        if logs is not None:
            logs.append(entry)
        print(entry, file=sys.stdout, flush=True)
        return entry

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
        if metadata_columns:
            working["metadata"] = working[metadata_columns].to_dict(orient="records")
        else:
            working["metadata"] = [{} for _ in range(len(working))]
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

    def _summarize_notebook_failure(self, result: Any) -> str:
        attempts = getattr(result, "attempts", []) or []
        if attempts:
            last_attempt = attempts[-1]
            if last_attempt.get("error_message"):
                return str(last_attempt["error_message"])
            notes = last_attempt.get("notes") or []
            if notes:
                return " | ".join(str(note) for note in notes)
        notes = getattr(result, "notes", []) or []
        if notes:
            return " | ".join(str(note) for note in notes)
        logs = getattr(result, "logs", []) or []
        if logs:
            return str(logs[-1])
        return "unknown notebook generation error"
