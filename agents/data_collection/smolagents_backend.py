import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import httpx
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.agents import RunResult
from smolagents.local_python_executor import BASE_BUILTIN_MODULES
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import DockerExecutor

from agents.tools import build_search_tools


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
DOCKERFILE_PATH = Path(__file__).with_name("Dockerfile")
AUTHORIZED_IMPORTS = [
    "aiohttp",
    "asyncio",
    "black",
    "bs4",
    "click",
    "colorama",
    "dateutil",
    "ddgs",
    "dotenv",
    "fake_useragent",
    "fastapi",
    "html",
    "html5lib",
    "httpx",
    "ipykernel",
    "IPython",
    "jupyter_client",
    "jupyter_kernel_gateway",
    "json",
    "loguru",
    "lxml",
    "lxml.html",
    "markdownify",
    "mypy",
    "numpy",
    "openpyxl",
    "orjson",
    "os",
    "pandas",
    "pathlib",
    "parsel",
    "pdfplumber",
    "PIL",
    "playwright",
    "playwright.async_api",
    "playwright.sync_api",
    "pydantic",
    "pypdf",
    "pytest",
    "pytest_asyncio",
    "pytz",
    "rapidfuzz",
    "readability",
    "regex",
    "requests",
    "rich",
    "ruff",
    "scrapy",
    "selenium",
    "selenium.webdriver",
    "selenium.webdriver.chrome",
    "selenium.webdriver.chrome.options",
    "selenium.webdriver.chrome.service",
    "selenium.webdriver.common",
    "selenium.webdriver.common.by",
    "selenium.webdriver.common.keys",
    "selenium.webdriver.support",
    "selenium.webdriver.support.expected_conditions",
    "selenium.webdriver.support.ui",
    "selenium_stealth",
    "selectolax",
    "selectolax.parser",
    "tenacity",
    "trafilatura",
    "typer",
    "uvicorn",
    "urllib",
    "urllib.parse",
    "xlsxwriter",
    "yaml",
]
DEFAULT_SAFE_IMPORTS = sorted(BASE_BUILTIN_MODULES)
EFFECTIVE_AUTHORIZED_IMPORTS = sorted(set(DEFAULT_SAFE_IMPORTS) | set(AUTHORIZED_IMPORTS))


@dataclass(slots=True)
class SmolagentsCollectionResult:
    records: list[dict[str, Any]]
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


class PrebakedDockerExecutor(DockerExecutor):
    """Docker executor that relies on the image contents instead of runtime pip installs."""

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        if additional_imports and hasattr(self, "logger"):
            self.logger.log(
                "Skipping runtime package installation; relying on preinstalled sandbox packages.",
                level=LogLevel.INFO,
            )
        return list(additional_imports)


class SmolagentsCollectionBackend:
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        self.llm_config = dict(llm_config or {})
        self.model = model or self._build_model(self.llm_config)
        self.tools = build_search_tools()

    def collect(self, source: Mapping[str, Any]) -> SmolagentsCollectionResult:
        source_type = str(source["type"])
        max_attempts = max(1, int(source.get("max_attempts", 2)))
        retry_on_empty = bool(source.get("retry_on_empty", source_type == "scrape"))
        max_steps = max(1, int(source.get("max_steps", 20)))
        logs: list[str] = []
        attempts: list[dict[str, Any]] = []
        base_task = self._build_task(source)

        for attempt_number in range(1, max_attempts + 1):
            task = self._build_attempt_task(base_task, attempts[-1] if attempts else None)

            try:
                logger = AgentLogger(level=LogLevel.ERROR)
                executor = PrebakedDockerExecutor(
                    additional_imports=AUTHORIZED_IMPORTS,
                    logger=logger,
                    **self._build_executor_kwargs(source),
                )
                with CodeAgent(
                    tools=self.tools,
                    model=self.model,
                    executor=executor,
                    executor_type="docker",
                    additional_authorized_imports=AUTHORIZED_IMPORTS,
                    max_steps=max_steps,
                    verbosity_level=1,
                ) as agent:
                    run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
                assert isinstance(run_result, RunResult)
                raw_output = "" if run_result.output is None else str(run_result.output)
                records, notes = self._parse_output(raw_output)
                attempt = {
                    "attempt": attempt_number,
                    "success": bool(records) or not retry_on_empty,
                    "row_count": len(records),
                    "output": raw_output,
                    "state": run_result.state,
                    "notes": notes,
                    "steps": run_result.steps or [],
                }
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(source, attempt))
                if attempt["success"]:
                    return SmolagentsCollectionResult(
                        records=records,
                        success=True,
                        logs=logs,
                        attempts=attempts,
                    )
            except Exception as error:
                attempt = {
                    "attempt": attempt_number,
                    "success": False,
                    "row_count": 0,
                    "output": "",
                    "state": "error",
                    "notes": [str(error)],
                    "steps": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(source, attempt))

        return SmolagentsCollectionResult(records=[], success=False, logs=logs, attempts=attempts)

    def _build_model(self, llm_config: Mapping[str, Any]) -> Any:
        provider = str(llm_config.get("provider", "openai_compatible")).lower()
        if provider in {"huggingface", "hf_inference", "inference_client"}:
            return InferenceClientModel(
                model_id=str(llm_config.get("model", "Qwen/Qwen3-Next-80B-A3B-Thinking")),
                provider=llm_config.get("hf_provider"),
                token=llm_config.get("api_key") or os.getenv("HF_TOKEN"),
                timeout=int(llm_config.get("timeout", 120)),
                base_url=llm_config.get("base_url"),
            )

        model_id = str(llm_config.get("model", "kimi-k2.5:cloud"))
        api_base = self._normalize_api_base(
            str(
                llm_config.get("api_base")
                or llm_config.get("base_url")
                or "http://localhost:11434/v1"
            )
        )
        api_key = str(llm_config.get("api_key") or os.getenv("OPENAI_API_KEY") or "ollama")
        client_kwargs = {
            "http_client": httpx.Client(trust_env=False),
        }
        return OpenAIModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            client_kwargs=client_kwargs,
            temperature=float(llm_config.get("temperature", 0.1)),
            max_tokens=int(llm_config.get("max_tokens", 4000)),
        )

    def _build_executor_kwargs(self, source: Mapping[str, Any]) -> dict[str, Any]:
        sandbox_config = dict(self.llm_config.get("sandbox", {}))
        cpu_limit = float(sandbox_config.get("cpu_limit", 0.5))
        container_run_kwargs = {
            "mem_limit": str(sandbox_config.get("memory_limit", "512m")),
            "cpu_quota": int(cpu_limit * 100000),
            "pids_limit": int(sandbox_config.get("pids_limit", 100)),
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "environment": self._build_container_env(),
        }
        if source.get("allow_network") is False:
            container_run_kwargs["network_disabled"] = True

        return {
            "host": str(sandbox_config.get("host", "127.0.0.1")),
            "port": int(sandbox_config.get("port", 8888)),
            "image_name": str(sandbox_config.get("image_name", "data-agent-smolagents-sandbox")),
            "build_new_image": bool(sandbox_config.get("build_new_image", False)),
            "container_run_kwargs": container_run_kwargs,
            "dockerfile_content": self._load_dockerfile_content(),
        }

    def _build_task(self, source: Mapping[str, Any]) -> str:
        source_json = json.dumps(dict(source), indent=2, ensure_ascii=False)
        limit = max(1, int(source.get("limit", 50)))
        research_instruction = (
            "If the extraction pattern is unclear, the site/API is unfamiliar, or an earlier approach fails, "
            "call `github_code_search` to look for existing implementation patterns or prior art and summarize "
            "the useful finding in `notes`. Use `web_search` only if GitHub results are insufficient."
        )
        output_instruction = (
            "Return the final answer as a JSON string, not a Python dict repr. "
            "Use `final_answer(json.dumps({...}, ensure_ascii=False))` exactly once. "
            "The top-level payload must be a JSON object with a `records` list and optional `notes` list."
        )
        scrape_output_instruction = (
            "For scrape sources, final records must be extracted from the target page itself, not from search engine "
            "results, GitHub results, or tool snippets. Search tools are only for research. They are never an "
            "acceptable final dataset.\n"
            "Each final record should include `text` containing the actual human-readable content extracted from the "
            "target page, such as a problem statement, post body, article text, or other page-derived content. "
            "Do not use search-result snippets as `text`."
        )
        scrape_decomposition_instruction = (
            "If the target page is an index, listing, archive, or landing page that links to child task documents "
            "(for example PDFs, problem pages, grade-specific pages, or answer sheets), do not stop at cataloging "
            "those links. Treat the page as a container and sub-parse the linked task resources where feasible.\n"
            "Prefer the most atomic useful record shape available:\n"
            "- best: one record per actual task/problem extracted from a linked page or PDF\n"
            "- acceptable fallback: one record per linked task document with clear metadata such as subject, grade, "
            "asset_type, and asset_url\n"
            "- avoid: one broad summary row for an entire subject or archive page when deeper parsing is possible\n"
            "When linked PDFs are present, prefer parsing them with `pypdf` or `pdfplumber` if that yields task text. "
            "Keep parent-child provenance in metadata, such as `parent_url`, `subject`, `grade`, `asset_type`, and "
            "`source_url`."
        )
        if source["type"] == "scrape":
            return (
                "Collect records from the configured webpage.\n"
                f"Source configuration:\n{source_json}\n\n"
                f"Target: extract up to {limit} useful records from this source and return them in the expected output format.\n"
                f"{research_instruction}\n"
                f"{output_instruction}\n"
                f"{scrape_output_instruction}\n"
                f"{scrape_decomposition_instruction}"
            )
        if source["type"] == "api":
            return (
                "Collect records from the configured API.\n"
                f"Source configuration:\n{source_json}\n\n"
                f"Target: extract up to {limit} useful records from this source and return them in the expected output format.\n"
                f"{research_instruction}\n"
                f"{output_instruction}"
            )
        raise ValueError(f"Unsupported smolagents source type '{source['type']}'.")

    def _build_attempt_task(
        self,
        base_task: str,
        previous_attempt: Mapping[str, Any] | None,
    ) -> str:
        if previous_attempt is None:
            return base_task
        return (
            f"{base_task}\n\n"
            "Previous attempt was not acceptable.\n"
            f"Previous output:\n{previous_attempt.get('output') or '<empty>'}\n"
            f"Previous notes: {json.dumps(previous_attempt.get('notes', []), ensure_ascii=False)}\n"
            "Revise the extraction code materially. If the failure suggests a missing pattern or unfamiliar API, "
            "use `github_code_search` to find a better existing implementation approach. Use `web_search` only if "
            "GitHub results are insufficient, and still return strict JSON via "
            "`final_answer(json.dumps({...}, ensure_ascii=False))`."
        )

    def _parse_output(self, raw_output: Any) -> tuple[list[dict[str, Any]], list[str]]:
        payload = self._parse_json_payload(raw_output)
        if isinstance(payload, list):
            return [_coerce_record(record) for record in payload], []
        if not isinstance(payload, dict):
            raise ValueError("Agent output was not a JSON object or array.")

        raw_records = payload.get("records", [])
        if isinstance(raw_records, dict):
            raw_records = [raw_records]
        if not isinstance(raw_records, list):
            raise ValueError("Agent output field `records` must be a list.")

        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        if not isinstance(notes, list):
            notes = [str(notes)]

        records = [_coerce_record(record) for record in raw_records]
        return records, [str(note) for note in notes]

    @staticmethod
    def _parse_json_payload(raw_output: Any) -> Any:
        if isinstance(raw_output, (dict, list)):
            return raw_output

        text = str(raw_output).strip()
        if not text:
            raise ValueError("Agent returned an empty output.")

        candidates = [text]
        match = JSON_BLOCK_PATTERN.search(text)
        if match:
            candidates.append(match.group(1).strip())

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
            except TypeError:
                continue

        for candidate in candidates:
            try:
                return ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        raise ValueError(f"Agent output is not valid JSON: {text[:500]}")

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        normalized = api_base.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized[: -len("/chat/completions")]
        return normalized

    @staticmethod
    def _format_attempt_logs(source: Mapping[str, Any], attempt: Mapping[str, Any]) -> list[str]:
        source_key = SmolagentsCollectionBackend._source_key(source)
        status = "succeeded" if attempt["success"] else "failed"
        lines = [
            f"smolagents attempt {attempt['attempt']} for {source_key} {status} with {attempt['row_count']} rows.",
        ]
        if attempt.get("error_message"):
            lines.append(
                f"smolagents attempt {attempt['attempt']} error for {source_key}: {attempt['error_message']}"
            )
        notes = attempt.get("notes") or []
        if notes:
            lines.append(
                f"smolagents attempt {attempt['attempt']} notes for {source_key}: "
                + " | ".join(str(note) for note in notes)
            )
        return lines

    @staticmethod
    def _source_key(source: Mapping[str, Any]) -> str:
        return str(source.get("name") or source.get("url") or source.get("endpoint") or source["type"])

    def _build_container_env(self) -> dict[str, str]:
        environment: dict[str, str] = {}
        hf_token = self.llm_config.get("hf_token") or self.llm_config.get("api_key") or os.getenv("HF_TOKEN")
        if hf_token:
            environment["HF_TOKEN"] = str(hf_token)

        github_config = self.llm_config.get("github", {})
        github_token = None
        if isinstance(github_config, Mapping):
            github_token = github_config.get("token") or github_config.get("api_key")
        github_token = (
            github_token
            or self.llm_config.get("github_token")
            or self.llm_config.get("gh_token")
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("GH_TOKEN")
        )
        if github_token:
            environment["GITHUB_TOKEN"] = str(github_token)
        return environment

    @staticmethod
    def _load_dockerfile_content() -> str:
        return DOCKERFILE_PATH.read_text(encoding="utf-8")


def _coerce_record(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return {"text": str(value)}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
