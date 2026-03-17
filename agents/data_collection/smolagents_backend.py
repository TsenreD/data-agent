import ast
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import httpx
import pandas as pd
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.agents import RunResult
from smolagents.local_python_executor import BASE_BUILTIN_MODULES
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.remote_executors import DockerExecutor

from agents.tools import build_search_tools

from .skillset import load_skill


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = Path(__file__).with_name("Dockerfile")
EDA_DOCKERFILE_PATH = Path(__file__).with_name("EdaDockerfile")
DEFAULT_SANDBOX_MEMORY_LIMIT = "512m"
DEFAULT_SANDBOX_SHM_SIZE = "1g"
DEFAULT_SANDBOX_CPU_LIMIT = 0.5
DEFAULT_SANDBOX_PIDS_LIMIT = 100
AUTHORIZED_IMPORTS = [
    "aiohttp",
    "agents",
    "agents.data_collection",
    "agents.data_collection.pdf_utils",
    "agents.data_collection.web_utils",
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
    "fitz",
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
    "pdfminer",
    "pdfminer.high_level",
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
    "seaborn",
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
PDF_EXTRACTION_HELPER = (
    "For PDFs, import and use the deterministic helper instead of writing a parser from scratch:\n"
    "`from agents.data_collection.pdf_utils import extract_pdf_text_from_url`\n"
    "This helper already tries `PyMuPDF` (`fitz`) first, then `pdfplumber`, then "
    "`pdfminer.high_level.extract_text`, and returns a dict with `method`, `text`, and `page_texts`.\n"
    "If it returns `method == \"unparsed\"` or nearly empty text, treat the PDF as likely scanned/image-based and "
    "report that clearly in `notes` instead of pretending extraction succeeded."
)
SITE_EXTRACTION_HELPER = (
    "For normal webpages, import and use the deterministic helpers from "
    "`agents.data_collection.web_utils` instead of inventing a parser stack from scratch.\n"
    "Available helpers:\n"
    "- `fetch_html(url)`: requests-based fetch with stable headers\n"
    "- `extract_main_text(html, url=None)`: tries `trafilatura`, then falls back to `BeautifulSoup`\n"
    "- `extract_links(html, base_url, ...)`: normalized link extraction with `BeautifulSoup`\n"
    "- `fetch_rendered_html_selenium(url)`: browser-rendered fallback for JS-heavy pages\n"
    "- `fetch_and_extract(url, use_selenium=False)`: fetch + main-text extraction + link extraction\n"
    "Recommended order:\n"
    "1. try `fetch_and_extract(url)` for ordinary pages\n"
    "2. if text is sparse, selectors are missing, or content is JS-rendered, try `fetch_and_extract(url, use_selenium=True)`\n"
    "3. use `extract_links(...)` to expand archive/index pages into child pages before shaping final records"
)
EDA_NOTEBOOK_IMPORTS = [
    "IPython",
    "IPython.display",
    "collections",
    "json",
    "math",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "pathlib",
    "re",
    "seaborn",
    "statistics",
]
NOTEBOOK_ALLOWED_IMPORT_ROOTS = {
    "IPython",
    "collections",
    "json",
    "math",
    "matplotlib",
    "numpy",
    "pandas",
    "pathlib",
    "re",
    "seaborn",
    "statistics",
}
DEFAULT_SAFE_IMPORTS = sorted(BASE_BUILTIN_MODULES)
EFFECTIVE_AUTHORIZED_IMPORTS = sorted(set(DEFAULT_SAFE_IMPORTS) | set(AUTHORIZED_IMPORTS))
EFFECTIVE_EDA_NOTEBOOK_IMPORTS = sorted(set(DEFAULT_SAFE_IMPORTS) | set(EDA_NOTEBOOK_IMPORTS))
NOTEBOOK_BOOTSTRAP_MARKER = "# data-agent notebook bootstrap"
NOTEBOOK_BOOTSTRAP_SOURCE = """# data-agent notebook bootstrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)
"""


@dataclass(slots=True)
class SmolagentsCollectionResult:
    records: list[dict[str, Any]]
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class NotebookGenerationResult:
    notebook: dict[str, Any] | None
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DatasetInspectionResult:
    summary: dict[str, Any] | None
    success: bool
    logs: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PrebakedDockerExecutor(DockerExecutor):
    """Docker executor that relies on the image contents instead of runtime pip installs."""

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        if additional_imports and hasattr(self, "logger"):
            self.logger.log(
                "Skipping runtime package installation; relying on preinstalled sandbox packages.",
                level=LogLevel.INFO,
            )
        return list(additional_imports)


class _SmolagentsDockerBackendBase:
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
        *,
        sandbox_key: str = "sandbox",
        dockerfile_path: Path = DOCKERFILE_PATH,
        default_image_name: str = "data-agent-smolagents-sandbox",
        default_port: int = 8888,
    ) -> None:
        self.llm_config = dict(llm_config or {})
        self.agent_config = dict(agent_config or {})
        self.model = model or self._build_model(self.llm_config)
        self.sandbox_key = sandbox_key
        self.dockerfile_path = dockerfile_path
        self.default_image_name = default_image_name
        self.default_port = default_port

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

    def _build_executor_kwargs(self, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
        source = source or {}
        sandbox_config = dict(self.llm_config.get(self.sandbox_key, {}))
        container_run_kwargs = self._build_container_run_kwargs(source, sandbox_config)

        return {
            "host": str(sandbox_config.get("host", "127.0.0.1")),
            "port": int(sandbox_config.get("port", self.default_port)),
            "image_name": str(sandbox_config.get("image_name", self.default_image_name)),
            "build_new_image": bool(sandbox_config.get("build_new_image", False)),
            "container_run_kwargs": container_run_kwargs,
            "dockerfile_content": self._load_dockerfile_content(),
        }

    def _build_container_run_kwargs(
        self,
        source: Mapping[str, Any],
        sandbox_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        container_run_kwargs = {
            **self._build_resource_limits(sandbox_config),
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "environment": self._build_container_env(),
        }
        if source.get("allow_network") is False:
            container_run_kwargs["network_disabled"] = True
        return container_run_kwargs

    @staticmethod
    def _build_resource_limits(sandbox_config: Mapping[str, Any]) -> dict[str, Any]:
        cpu_limit = float(sandbox_config.get("cpu_limit", DEFAULT_SANDBOX_CPU_LIMIT))
        return {
            "mem_limit": str(sandbox_config.get("memory_limit", DEFAULT_SANDBOX_MEMORY_LIMIT)),
            "cpu_quota": int(cpu_limit * 100000),
            "pids_limit": int(sandbox_config.get("pids_limit", DEFAULT_SANDBOX_PIDS_LIMIT)),
            # Chrome and Chromium routinely crash in Docker with the default 64 MB /dev/shm.
            "shm_size": str(sandbox_config.get("shm_size", DEFAULT_SANDBOX_SHM_SIZE)),
        }

    def _run_agent(
        self,
        *,
        task: str,
        instructions: str,
        tools: list[Any],
        additional_imports: list[str],
        max_steps: int,
        source: Mapping[str, Any] | None = None,
    ) -> RunResult:
        logger = AgentLogger(level=LogLevel.ERROR)
        executor = PrebakedDockerExecutor(
            additional_imports=additional_imports,
            logger=logger,
            **self._build_executor_kwargs(source),
        )
        with CodeAgent(
            tools=tools,
            model=self.model,
            executor=executor,
            executor_type="docker",
            additional_authorized_imports=additional_imports,
            max_steps=max_steps,
            verbosity_level=1,
            instructions=instructions,
        ) as agent:
            run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
        assert isinstance(run_result, RunResult)
        return run_result

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        normalized = api_base.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized[: -len("/chat/completions")]
        return normalized

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

    def _load_dockerfile_content(self) -> str:
        return self.dockerfile_path.read_text(encoding="utf-8")


class SmolagentsCollectionBackend(_SmolagentsDockerBackendBase):
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(
            llm_config=llm_config,
            agent_config=agent_config,
            model=model,
            sandbox_key="sandbox",
            dockerfile_path=DOCKERFILE_PATH,
            default_image_name="data-agent-smolagents-sandbox",
            default_port=8888,
        )
        self.tools = build_search_tools()
        self.instructions = load_skill("data_collection")

    def _build_container_run_kwargs(
        self,
        source: Mapping[str, Any],
        sandbox_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        container_run_kwargs = super()._build_container_run_kwargs(source, sandbox_config)
        mount_host_path = str(source.get("mount_host_path") or PROJECT_ROOT)
        mount_container_path = str(source.get("mount_container_path") or "/workspace")
        volumes = container_run_kwargs.get("volumes", {})
        volumes[mount_host_path] = {"bind": mount_container_path, "mode": "ro"}
        container_run_kwargs["volumes"] = volumes
        container_run_kwargs["working_dir"] = "/workspace"
        return container_run_kwargs

    def collect(self, source: Mapping[str, Any]) -> SmolagentsCollectionResult:
        source_type = str(source["type"])
        max_attempts = max(1, int(source.get("max_attempts", self.agent_config.get("max_attempts", 2))))
        retry_on_empty = bool(source.get("retry_on_empty", self.agent_config.get("retry_on_empty", source_type == "scrape")))
        max_steps = max(1, int(source.get("max_steps", self.agent_config.get("max_steps", 20))))
        logs: list[str] = []
        attempts: list[dict[str, Any]] = []
        base_task = self._build_task(source, max_steps=max_steps, max_attempts=max_attempts)

        for attempt_number in range(1, max_attempts + 1):
            task = self._build_attempt_task(base_task, attempts[-1] if attempts else None)

            try:
                run_result = self._run_agent(
                    task=task,
                    instructions=self.instructions,
                    tools=self.tools,
                    additional_imports=AUTHORIZED_IMPORTS,
                    max_steps=max_steps,
                    source=source,
                )
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

    def _build_task(self, source: Mapping[str, Any], *, max_steps: int, max_attempts: int) -> str:
        source_json = json.dumps(dict(source), indent=2, ensure_ascii=False)
        limit = max(1, int(source.get("limit", 50)))
        extraction_instructions = str(source.get("extraction_instructions", "")).strip()
        budget_instruction = (
            f"You have at most {max_steps} agent steps in this attempt. "
            "Plan accordingly and return the final answer as soon as you have enough data. "
            f"The system may retry you up to {max_attempts} attempts total if this attempt fails."
        )
        research_instruction = (
            "If the extraction pattern is unclear, the site/API is unfamiliar, or an earlier approach fails, "
            "call `github_code_search` to look for existing implementation patterns or prior art and summarize "
            "the useful finding in `notes`. Use `web_search` only if GitHub results are insufficient."
        )
        scrape_strategy_instruction = (
            "For scrape sources, follow this order strictly: "
            "1) try the provided helper modules first, "
            "2) if they fail or are clearly insufficient, research with `github_code_search` first and `web_search` second, "
            "3) only then write custom parsing code, keeping it narrow and source-specific."
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
        scrape_fetch_instruction = (
            "For ordinary HTML pages, prefer the provided site helper module before writing custom parsing code.\n"
            f"{SITE_EXTRACTION_HELPER}"
        )
        extraction_guidance = (
            "User extraction instructions:\n"
            f"{extraction_instructions}\n"
            if extraction_instructions
            else ""
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
            "When linked PDFs are present, prefer parsing them with `PyMuPDF` (`fitz`) first, then `pdfplumber`, then "
            "`pdfminer.high_level.extract_text`. Do not invent a brand-new PDF parser if the provided helper is enough.\n"
            f"{PDF_EXTRACTION_HELPER}\n"
            "Keep parent-child provenance in metadata, such as `parent_url`, `subject`, `grade`, `asset_type`, and "
            "`source_url`."
        )
        if source["type"] == "scrape":
            return (
                "Collect records from the configured webpage.\n"
                f"Source configuration:\n{source_json}\n\n"
                f"Target: extract up to {limit} useful records from this source and return them in the expected output format.\n"
                f"{extraction_guidance}"
                f"{budget_instruction}\n"
                f"{scrape_strategy_instruction}\n"
                f"{research_instruction}\n"
                f"{output_instruction}\n"
                f"{scrape_output_instruction}\n"
                f"{scrape_fetch_instruction}\n"
                f"{scrape_decomposition_instruction}"
            )
        if source["type"] == "api":
            return (
                "Collect records from the configured API.\n"
                f"Source configuration:\n{source_json}\n\n"
                f"Target: extract up to {limit} useful records from this source and return them in the expected output format.\n"
                f"{budget_instruction}\n"
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
        payload = _parse_json_payload(raw_output)
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


class SmolagentsNotebookBackend(_SmolagentsDockerBackendBase):
    def __init__(
        self,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(
            llm_config=llm_config,
            agent_config=agent_config,
            model=model,
            sandbox_key="eda_sandbox",
            dockerfile_path=EDA_DOCKERFILE_PATH,
            default_image_name="data-agent-eda-sandbox",
            default_port=8890,
        )
        self.inspection_instructions = load_skill("eda_inspection")
        self.instructions = load_skill("eda_notebook")

    def _build_container_run_kwargs(
        self,
        source: Mapping[str, Any],
        sandbox_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        container_run_kwargs = super()._build_container_run_kwargs(source, sandbox_config)
        mount_host_path = str(source.get("mount_host_path") or PROJECT_ROOT)
        mount_container_path = str(source.get("mount_container_path") or "/workspace")
        volumes = container_run_kwargs.get("volumes", {})
        volumes[mount_host_path] = {"bind": mount_container_path, "mode": "rw"}
        container_run_kwargs["volumes"] = volumes
        container_run_kwargs["working_dir"] = "/workspace"
        environment = dict(container_run_kwargs.get("environment", {}))
        environment.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        container_run_kwargs["environment"] = environment
        return container_run_kwargs

    def generate_notebook(
        self,
        frame: pd.DataFrame,
        dataset_path: str | Path,
        notebook_path: str | Path,
    ) -> NotebookGenerationResult:
        dataset_path = Path(dataset_path)
        notebook_path = Path(notebook_path)
        logs: list[str] = []
        attempts: list[dict[str, Any]] = []
        generation_attempts: list[dict[str, Any]] = []
        max_attempts = max(1, int(self.agent_config.get("max_attempts", 2)))
        max_steps = max(1, int(self.agent_config.get("max_steps", 12)))
        inspection = self.inspect_dataset(dataset_path)
        logs.extend(inspection.logs)
        attempts.extend(inspection.attempts)
        if not inspection.success or inspection.summary is None:
            return NotebookGenerationResult(
                notebook=None,
                success=False,
                logs=logs,
                attempts=attempts,
                notes=inspection.notes,
            )

        base_task = self._build_task(
            frame,
            dataset_path,
            notebook_path,
            inspection.summary,
            max_steps=max_steps,
            max_attempts=max_attempts,
        )

        for attempt_number in range(1, max_attempts + 1):
            task = self._build_attempt_task(base_task, generation_attempts[-1] if generation_attempts else None)

            try:
                with CodeAgent(
                    tools=[],
                    model=self.model,
                    additional_authorized_imports=EDA_NOTEBOOK_IMPORTS,
                    max_steps=max_steps,
                    verbosity_level=1,
                    instructions=self.instructions,
                ) as agent:
                    run_result = agent.run(task, max_steps=max_steps, return_full_result=True)
                assert isinstance(run_result, RunResult)
                raw_output = "" if run_result.output is None else str(run_result.output)
                notebook, notes = self._parse_output(raw_output)
                normalized = self._normalize_notebook(notebook)
                self._validate_notebook(normalized)
                attempt = {
                    "phase": "notebook",
                    "attempt": attempt_number,
                    "success": True,
                    "output": raw_output,
                    "state": run_result.state,
                    "notes": notes,
                    "steps": run_result.steps or [],
                }
                generation_attempts.append(attempt)
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(notebook_path, attempt))
                return NotebookGenerationResult(
                    notebook=normalized,
                    success=True,
                    logs=logs,
                    attempts=attempts,
                    notes=inspection.notes + notes,
                )
            except Exception as error:
                attempt = {
                    "phase": "notebook",
                    "attempt": attempt_number,
                    "success": False,
                    "output": "",
                    "state": "error",
                    "notes": [str(error)],
                    "steps": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
                generation_attempts.append(attempt)
                attempts.append(attempt)
                logs.extend(self._format_attempt_logs(notebook_path, attempt))

        return NotebookGenerationResult(
            notebook=None,
            success=False,
            logs=logs,
            attempts=attempts,
            notes=[str(attempts[-1].get("error_message", ""))] if attempts else [],
        )

    def inspect_dataset(self, dataset_path: str | Path) -> DatasetInspectionResult:
        dataset_path = Path(dataset_path)
        mount_host_path, mount_container_path, sandbox_dataset_path = self._sandbox_mount(dataset_path)
        logs: list[str] = []
        attempts: list[dict[str, Any]] = []
        max_attempts = max(1, int(self.agent_config.get("inspection_max_attempts", 1)))

        for attempt_number in range(1, max_attempts + 1):
            try:
                raw_output = self._run_inspection_script(
                    dataset_path=dataset_path,
                    mount_host_path=mount_host_path,
                    mount_container_path=mount_container_path,
                )
                summary, notes = self._parse_inspection_output(raw_output)
                attempt = {
                    "phase": "inspection",
                    "attempt": attempt_number,
                    "success": True,
                    "output": raw_output,
                    "state": "done",
                    "notes": notes,
                    "steps": [],
                }
                attempts.append(attempt)
                logs.extend(self._format_inspection_logs(dataset_path, attempt))
                return DatasetInspectionResult(
                    summary=summary,
                    success=True,
                    logs=logs,
                    attempts=attempts,
                    notes=notes,
                )
            except Exception as error:
                attempt = {
                    "phase": "inspection",
                    "attempt": attempt_number,
                    "success": False,
                    "output": "",
                    "state": "error",
                    "notes": [str(error)],
                    "steps": [],
                    "error_message": f"{type(error).__name__}: {error}",
                }
                attempts.append(attempt)
                logs.extend(self._format_inspection_logs(dataset_path, attempt))

        return DatasetInspectionResult(
            summary=None,
            success=False,
            logs=logs,
            attempts=attempts,
            notes=[str(attempts[-1].get("error_message", ""))] if attempts else [],
        )

    def _build_task(
        self,
        frame: pd.DataFrame,
        dataset_path: Path,
        notebook_path: Path,
        inspection_summary: Mapping[str, Any],
        *,
        max_steps: int,
        max_attempts: int,
    ) -> str:
        relative_dataset_path = Path(os.path.relpath(dataset_path, notebook_path.parent)).as_posix()
        preview_json = frame.head(min(3, len(frame))).to_json(orient="records", force_ascii=False, date_format="iso")
        schema_json = json.dumps({column: str(dtype) for column, dtype in frame.dtypes.items()}, ensure_ascii=False)
        non_null_json = json.dumps(
            {column: int(frame[column].notna().sum()) for column in frame.columns},
            ensure_ascii=False,
        )
        inspection_json = json.dumps(dict(inspection_summary), ensure_ascii=False)
        allowed_imports = ", ".join(sorted(NOTEBOOK_ALLOWED_IMPORT_ROOTS))
        return (
            "Generate an executable Jupyter notebook for exploratory data analysis of the unified dataset.\n"
            f"Notebook output path: {notebook_path.as_posix()}\n"
            f"Notebook dataset path: {relative_dataset_path}\n"
            f"Row count: {len(frame)}\n"
            f"Schema: {schema_json}\n"
            f"Non-null counts: {non_null_json}\n"
            f"Preview rows: {preview_json}\n\n"
            f"Observed dataset inspection summary: {inspection_json}\n\n"
            f"You have at most {max_steps} agent steps in this notebook-generation attempt. "
            "Use them to produce the final notebook directly instead of over-exploring. "
            f"The system may retry you up to {max_attempts} attempts total if this attempt fails.\n"
            "Return the final answer as a JSON string with top-level fields `notebook` and optional `notes`.\n"
            "Call `final_answer(json.dumps(payload, ensure_ascii=False))` exactly once.\n"
            "Do not print the notebook JSON instead of returning it.\n"
            "The `notebook` value must be a valid nbformat 4 notebook object.\n"
            "The notebook must contain at most 10 cells total.\n"
            "Every code cell must be executable later, with `execution_count` set to null and `outputs` set to an empty list.\n"
            f"Notebook code may only import these module roots: {allowed_imports}.\n"
            "The first code cell must define these aliases so later cells can rely on them: `Path`, `np`, `pd`, `plt`, `sns`, and `display`.\n"
            "Use this setup pattern in the first code cell: `from pathlib import Path`, `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`, `import seaborn as sns`, and a `try/except` import for `display` from `IPython.display`.\n"
            "Use the provided dataset path exactly in notebook code.\n"
            "Use the inspection summary to decide which sections deserve notebook space.\n"
            "Do not use shell escapes like `!pip`.\n"
            "Do not use IPython magics except `%matplotlib inline` if absolutely necessary.\n"
            "The notebook should include concise markdown sections and executable code for dataset loading, preview, schema summary, null analysis, label distribution when available, and text-length distributions when text is available.\n"
            "Code must handle missing columns gracefully and must not use shell commands, subprocesses, or arbitrary filesystem writes.\n"
            "Prefer a straightforward 5-7 cell notebook and never exceed 10 cells."
        )

    def _build_inspection_task(self, dataset_path: Path, sandbox_dataset_path: str) -> str:
        return (
            "Inspect the unified dataset and return a concise JSON summary of the real data.\n"
            f"Host dataset path: {dataset_path.as_posix()}\n"
            f"Sandbox dataset path: {sandbox_dataset_path}\n"
            "Load the dataset from the sandbox path with pandas and inspect the real rows and columns before answering.\n"
            "Return the final answer as a JSON string with top-level fields `summary` and optional `notes`.\n"
            "Keep `summary` compact, factual, and JSON-serializable."
        )

    def _run_inspection_script(
        self,
        *,
        dataset_path: Path,
        mount_host_path: Path,
        mount_container_path: str,
    ) -> str:
        sandbox_dataset_path = self._sandbox_dataset_path(dataset_path)
        script = f"""
import json
from collections import Counter

import pandas as pd

df = pd.read_json({sandbox_dataset_path!r}, lines=True)
summary = {{
    "row_count": int(len(df)),
    "columns": list(df.columns),
    "dtypes": {{column: str(dtype) for column, dtype in df.dtypes.items()}},
    "non_null_counts": {{column: int(df[column].notna().sum()) for column in df.columns}},
}}

recommended_sections = ["overview", "missingness"]

if "label" in df.columns and df["label"].notna().any():
    label_series = df["label"].dropna()
    label_summary = {{
        "unique_count": int(label_series.nunique()),
        "top_values": label_series.astype(str).value_counts().head(10).to_dict(),
    }}
    if pd.api.types.is_numeric_dtype(label_series):
        label_summary["describe"] = {{
            key: float(value) if key != "count" else int(value)
            for key, value in label_series.describe().to_dict().items()
        }}
    summary["label_summary"] = label_summary
    recommended_sections.append("label_distribution")

if "text" in df.columns and df["text"].notna().any():
    text_series = df["text"].fillna("").astype(str)
    text_lengths = text_series.str.split().str.len()
    summary["text_summary"] = {{
        "text_length_stats": {{
            "unit": "words",
            "mean": float(text_lengths.mean()),
            "median": float(text_lengths.median()),
            "p90": float(text_lengths.quantile(0.9)),
            "max": int(text_lengths.max()),
        }},
    }}
    recommended_sections.append("text_length_distribution")

if "source" in df.columns and df["source"].notna().any():
    summary["source_summary"] = df["source"].astype(str).value_counts().head(10).to_dict()
    recommended_sections.append("source_distribution")

metadata_keys = Counter()
for value in df.get("metadata", pd.Series(dtype=object)).dropna().head(50):
    if isinstance(value, dict):
        metadata_keys.update(str(key) for key in value.keys())
if metadata_keys:
    summary["metadata_key_sample"] = [key for key, _ in metadata_keys.most_common(10)]

summary["recommended_sections"] = list(dict.fromkeys(recommended_sections))

payload = {{
    "summary": summary,
    "notes": [
        "Inspected the real dataset inside the lightweight EDA sandbox.",
        "Summary is based on executed pandas inspection rather than static host assumptions.",
    ],
}}
print(json.dumps(payload, ensure_ascii=False))
"""
        return self._run_python_in_eda_sandbox(
            script=script,
            mount_host_path=mount_host_path,
            mount_container_path=mount_container_path,
        )

    def _run_python_in_eda_sandbox(
        self,
        *,
        script: str,
        mount_host_path: Path,
        mount_container_path: str,
    ) -> str:
        import docker

        sandbox_config = dict(self.llm_config.get(self.sandbox_key, {}))
        image_name = str(sandbox_config.get("image_name", self.default_image_name))
        build_new_image = bool(sandbox_config.get("build_new_image", False))
        client = docker.from_env()

        if not build_new_image:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                build_new_image = True

        if build_new_image:
            dockerfile_obj = BytesIO(self._load_dockerfile_content().encode("utf-8"))
            client.images.build(fileobj=dockerfile_obj, tag=image_name)

        environment = self._build_container_env()
        environment.setdefault("MPLCONFIGDIR", "/tmp/mpl")

        try:
            output = client.containers.run(
                image_name,
                command=["python3", "-c", script],
                remove=True,
                working_dir="/workspace",
                volumes={str(mount_host_path): {"bind": mount_container_path, "mode": "rw"}},
                environment=environment,
                **self._build_resource_limits(sandbox_config),
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                network_disabled=bool(sandbox_config.get("disable_network", True)),
            )
        except docker.errors.ContainerError as error:
            stderr = error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error)
            raise RuntimeError(f"EDA sandbox inspection failed: {stderr}") from error
        except docker.errors.DockerException as error:
            raise RuntimeError(f"EDA sandbox inspection failed: {error}") from error

        return output.decode("utf-8", errors="replace")

    def _build_attempt_task(
        self,
        base_task: str,
        previous_attempt: Mapping[str, Any] | None,
    ) -> str:
        if previous_attempt is None:
            return base_task
        return (
            f"{base_task}\n\n"
            "Previous notebook output was rejected.\n"
            f"Previous error: {previous_attempt.get('error_message') or 'unknown error'}\n"
            f"Previous notes: {json.dumps(previous_attempt.get('notes', []), ensure_ascii=False)}\n"
            f"Previous output:\n{previous_attempt.get('output') or '<empty>'}\n"
            "Revise the notebook structure materially and return strict JSON only."
        )

    def _build_inspection_attempt_task(
        self,
        base_task: str,
        previous_attempt: Mapping[str, Any] | None,
    ) -> str:
        if previous_attempt is None:
            return base_task
        return (
            f"{base_task}\n\n"
            "Previous inspection output was rejected.\n"
            f"Previous error: {previous_attempt.get('error_message') or 'unknown error'}\n"
            f"Previous notes: {json.dumps(previous_attempt.get('notes', []), ensure_ascii=False)}\n"
            f"Previous output:\n{previous_attempt.get('output') or '<empty>'}\n"
            "Inspect the dataset again and return a compact factual summary in strict JSON only."
        )

    def _parse_output(self, raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict):
            raise ValueError("Notebook agent output was not a JSON object.")
        notebook = payload.get("notebook")
        if isinstance(notebook, str):
            notebook = _parse_json_payload(notebook)
        if not isinstance(notebook, dict):
            raise ValueError("Notebook agent output did not include a notebook object.")
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        if not isinstance(notes, list):
            notes = [str(notes)]
        return notebook, [str(note) for note in notes]

    def _parse_inspection_output(self, raw_output: Any) -> tuple[dict[str, Any], list[str]]:
        payload = _parse_json_payload(raw_output)
        if not isinstance(payload, dict):
            raise ValueError("EDA inspection output was not a JSON object.")
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("EDA inspection output did not include a summary object.")
        notes = payload.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        if not isinstance(notes, list):
            notes = [str(notes)]
        return summary, [str(note) for note in notes]

    @staticmethod
    def _normalize_notebook(notebook: Mapping[str, Any]) -> dict[str, Any]:
        cells = notebook.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("Notebook must contain at least one cell.")
        if len(cells) > 10:
            raise ValueError(f"Notebook must contain at most 10 cells, got {len(cells)}.")

        normalized_cells: list[dict[str, Any]] = []
        for index, cell in enumerate(cells, start=1):
            if not isinstance(cell, Mapping):
                raise ValueError(f"Notebook cell {index} is not an object.")
            cell_type = str(cell.get("cell_type", "")).strip()
            if cell_type not in {"markdown", "code"}:
                raise ValueError(f"Notebook cell {index} has unsupported type '{cell_type}'.")
            normalized_cell = {
                "cell_type": cell_type,
                "metadata": dict(cell.get("metadata", {})),
                "source": _normalize_cell_source(cell.get("source", "")),
            }
            if cell_type == "code":
                normalized_cell["execution_count"] = None
                normalized_cell["outputs"] = []
            normalized_cells.append(normalized_cell)

        normalized_cells = SmolagentsNotebookBackend._ensure_notebook_bootstrap(normalized_cells)

        metadata = dict(notebook.get("metadata", {}))
        metadata.setdefault(
            "kernelspec",
            {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        )
        metadata.setdefault(
            "language_info",
            {
                "name": "python",
                "version": "3.11",
            },
        )
        return {
            "cells": normalized_cells,
            "metadata": metadata,
            "nbformat": int(notebook.get("nbformat", 4)),
            "nbformat_minor": int(notebook.get("nbformat_minor", 5)),
        }

    @staticmethod
    def _ensure_notebook_bootstrap(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            if NOTEBOOK_BOOTSTRAP_MARKER in source or _defines_required_notebook_aliases(source):
                return cells
            bootstrap = NOTEBOOK_BOOTSTRAP_SOURCE.rstrip()
            combined = f"{bootstrap}\n\n{source.lstrip()}" if source.strip() else bootstrap
            cell["source"] = _normalize_cell_source(combined)
            return cells

        if len(cells) >= 10:
            raise ValueError("Notebook must leave room for a bootstrap code cell with standard imports.")

        insertion_index = 1 if cells and cells[0].get("cell_type") == "markdown" else 0
        cells.insert(
            insertion_index,
            {
                "cell_type": "code",
                "metadata": {},
                "source": _normalize_cell_source(NOTEBOOK_BOOTSTRAP_SOURCE),
                "execution_count": None,
                "outputs": [],
            },
        )
        return cells

    @staticmethod
    def _validate_notebook(notebook: Mapping[str, Any]) -> None:
        if int(notebook.get("nbformat", 0)) != 4:
            raise ValueError("Notebook must use nbformat 4.")

        for index, cell in enumerate(notebook.get("cells", []), start=1):
            source = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            sanitized_source = _sanitize_notebook_source(source, index)
            try:
                tree = ast.parse(sanitized_source)
            except SyntaxError as error:
                raise ValueError(f"Notebook code cell {index} is not valid Python: {error}") from error
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        _validate_import_root(alias.name, index)
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        raise ValueError(f"Notebook code cell {index} uses relative imports, which are not allowed.")
                    module_name = node.module or ""
                    _validate_import_root(module_name, index)

    @staticmethod
    def _format_attempt_logs(notebook_path: Path, attempt: Mapping[str, Any]) -> list[str]:
        status = "succeeded" if attempt["success"] else "failed"
        lines = [f"EDA notebook attempt {attempt['attempt']} for {notebook_path} {status}."]
        if attempt.get("error_message"):
            lines.append(
                f"EDA notebook attempt {attempt['attempt']} error for {notebook_path}: {attempt['error_message']}"
            )
        notes = attempt.get("notes") or []
        if notes:
            lines.append(
                f"EDA notebook attempt {attempt['attempt']} notes for {notebook_path}: "
                + " | ".join(str(note) for note in notes)
            )
        return lines

    @staticmethod
    def _format_inspection_logs(dataset_path: Path, attempt: Mapping[str, Any]) -> list[str]:
        status = "succeeded" if attempt["success"] else "failed"
        lines = [f"EDA inspection attempt {attempt['attempt']} for {dataset_path} {status}."]
        if attempt.get("error_message"):
            lines.append(
                f"EDA inspection attempt {attempt['attempt']} error for {dataset_path}: {attempt['error_message']}"
            )
        notes = attempt.get("notes") or []
        if notes:
            lines.append(
                f"EDA inspection attempt {attempt['attempt']} notes for {dataset_path}: "
                + " | ".join(str(note) for note in notes)
            )
        return lines

    @staticmethod
    def _sandbox_dataset_path(dataset_path: Path) -> str:
        return SmolagentsNotebookBackend._sandbox_mount(dataset_path)[2]

    @staticmethod
    def _sandbox_mount(dataset_path: Path) -> tuple[Path, str, str]:
        resolved_dataset_path = dataset_path.resolve()
        resolved_project_root = PROJECT_ROOT.resolve()
        if resolved_dataset_path.is_relative_to(resolved_project_root):
            relative_path = resolved_dataset_path.relative_to(resolved_project_root)
            return PROJECT_ROOT, "/workspace", (Path("/workspace") / relative_path).as_posix()
        mount_host_path = resolved_dataset_path.parent
        mount_container_path = "/workspace/input"
        return mount_host_path, mount_container_path, (Path(mount_container_path) / resolved_dataset_path.name).as_posix()


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


def _normalize_cell_source(source: Any) -> list[str]:
    if isinstance(source, list):
        return [str(line) for line in source]
    text = str(source)
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if text and not lines:
        return [text]
    if text and not text.endswith(("\n", "\r")):
        lines[-1] = lines[-1]
    return lines


def _sanitize_notebook_source(source: str, cell_index: int) -> str:
    sanitized_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            sanitized_lines.append(line)
            continue
        if stripped.startswith("!"):
            raise ValueError(f"Notebook code cell {cell_index} uses shell escapes, which are not allowed.")
        if stripped.startswith("%%"):
            raise ValueError(f"Notebook code cell {cell_index} uses cell magics, which are not allowed.")
        if stripped.startswith("%"):
            if stripped == "%matplotlib inline":
                continue
            raise ValueError(f"Notebook code cell {cell_index} uses unsupported IPython magic '{stripped}'.")
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def _defines_required_notebook_aliases(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def collect_defined_names(statements: list[ast.stmt], defined_names: set[str]) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        defined_names.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, (ast.Try, ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                child_blocks = [node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])]
                if isinstance(node, ast.Try):
                    child_blocks.extend(handler.body for handler in node.handlers)
                for block in child_blocks:
                    collect_defined_names(block, defined_names)
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    collect_defined_names(case.body, defined_names)

    defined_names: set[str] = set()
    collect_defined_names(tree.body, defined_names)

    required_names = {"Path", "display", "np", "pd", "plt", "sns"}
    return required_names.issubset(defined_names)


def _validate_import_root(module_name: str, cell_index: int) -> None:
    root = module_name.split(".", 1)[0]
    if root not in NOTEBOOK_ALLOWED_IMPORT_ROOTS:
        raise ValueError(
            f"Notebook code cell {cell_index} imports '{module_name}', which is outside the approved EDA library set."
        )
