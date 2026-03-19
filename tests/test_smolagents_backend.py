import agents.data_collection.smolagents_backend as smolagents_backend_module
from agents.data_collection.smolagents_backend import (
    AUTHORIZED_IMPORTS,
    DEFAULT_SAFE_IMPORTS,
    DOCKERFILE_PATH,
    EDA_DOCKERFILE_PATH,
    EDA_NOTEBOOK_IMPORTS,
    EFFECTIVE_AUTHORIZED_IMPORTS,
    EFFECTIVE_EDA_NOTEBOOK_IMPORTS,
    PROJECT_ROOT,
    PrebakedDockerExecutor,
    SmolagentsCollectionBackend,
    SmolagentsNotebookBackend,
)
from smolagents.agents import RunResult


def test_normalize_api_base_strips_chat_completions_suffix() -> None:
    assert (
        SmolagentsCollectionBackend._normalize_api_base(
            "http://localhost:11434/v1/chat/completions"
        )
        == "http://localhost:11434/v1"
    )


def test_parse_output_accepts_plain_json() -> None:
    backend = SmolagentsCollectionBackend(model=object())
    records, notes = backend._parse_output('{"records":[{"text":"hello","title":"greeting"}],"notes":["ok"]}')

    assert records == [{"text": "hello", "title": "greeting"}]
    assert notes == ["ok"]


def test_parse_output_accepts_fenced_json() -> None:
    backend = SmolagentsCollectionBackend(model=object())
    records, notes = backend._parse_output('```json\n{"records":[{"text":"hello"}]}\n```')

    assert records == [{"text": "hello"}]
    assert notes == []


def test_parse_output_accepts_python_literal_dict_string() -> None:
    backend = SmolagentsCollectionBackend(model=object())
    records, notes = backend._parse_output("{'records': [{'text': 'hello'}], 'notes': ['ok']}")

    assert records == [{"text": "hello"}]
    assert notes == ["ok"]


def test_parse_output_accepts_direct_dict_output() -> None:
    backend = SmolagentsCollectionBackend(model=object())
    records, notes = backend._parse_output({"records": [{"text": "hello"}], "notes": ["ok"]})

    assert records == [{"text": "hello"}]
    assert notes == ["ok"]


def test_build_task_requires_github_search_before_writing_code() -> None:
    backend = SmolagentsCollectionBackend(model=object())

    task = backend._build_task(
        {
            "type": "scrape",
            "url": "https://example.com",
            "extraction_instructions": "Extract one record per problem and preserve source_url.",
        },
        max_steps=9,
        max_attempts=2,
    )

    assert "If the extraction pattern is unclear" in task
    assert "Use `web_search` only if GitHub results are insufficient." in task
    assert "not from search engine results" in task
    assert "Do not use search-result snippets as `text`." in task
    assert "final_answer(json.dumps" in task
    assert "If the target page is an index, listing, archive, or landing page" in task
    assert "do not stop at cataloging those links" in task
    assert "best: one record per actual task/problem" in task
    assert "You have at most 9 agent steps in this attempt." in task
    assert "follow this order strictly" in task
    assert "try the provided helper modules first" in task
    assert "extract_pdf_text_from_url" in task
    assert "agents.data_collection.pdf_utils" in task
    assert "fetch_and_extract(url)" in task
    assert "agents.data_collection.web_utils" in task
    assert "User extraction instructions:" in task
    assert "Extract one record per problem and preserve source_url." in task


def test_dockerfile_contains_local_requirements() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    requirements = [
        "aiohttp",
        "beautifulsoup4",
        "ddgs",
        "fastapi",
        "httpx",
        "markdownify",
        "numpy",
        "pandas",
        "pdfminer.six",
        "PyMuPDF",
        "playwright",
        "requests",
        "scrapy",
        "seaborn",
        "selenium",
        "selenium-stealth",
        "selectolax",
        "trafilatura",
    ]

    for requirement in requirements:
        assert requirement in content


def test_eda_dockerfile_contains_eda_requirements() -> None:
    content = EDA_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "matplotlib" in content
    assert "numpy" in content
    assert "pandas" in content
    assert "seaborn" in content


def test_dockerfile_contains_browser_runtime_dependencies() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "chromium" in content
    assert "chromium-driver" in content
    assert "playwright" in content
    assert "xvfb" in content
    assert "libgtk-3-0" in content


def test_prebaked_executor_skips_runtime_package_installation() -> None:
    executor = object.__new__(PrebakedDockerExecutor)
    installed = executor.install_packages(["requests", "bs4"])

    assert installed == ["requests", "bs4"]


def test_all_imports_are_authorized() -> None:
    assert "*" not in AUTHORIZED_IMPORTS
    assert "json" in AUTHORIZED_IMPORTS
    assert "os" in AUTHORIZED_IMPORTS
    assert "pathlib" in AUTHORIZED_IMPORTS
    assert "urllib.parse" in AUTHORIZED_IMPORTS
    assert "requests" in AUTHORIZED_IMPORTS
    assert "agents.data_collection.pdf_utils" in AUTHORIZED_IMPORTS
    assert "agents.data_collection.web_utils" in AUTHORIZED_IMPORTS
    assert "bs4" in AUTHORIZED_IMPORTS
    assert "fitz" in AUTHORIZED_IMPORTS
    assert "pdfminer.high_level" in AUTHORIZED_IMPORTS
    assert "playwright.sync_api" in AUTHORIZED_IMPORTS
    assert "selenium.webdriver.common.by" in AUTHORIZED_IMPORTS
    assert "yaml" in AUTHORIZED_IMPORTS


def test_effective_authorized_imports_include_smolagents_safe_defaults() -> None:
    assert "datetime" in DEFAULT_SAFE_IMPORTS
    assert "re" in DEFAULT_SAFE_IMPORTS
    assert set(DEFAULT_SAFE_IMPORTS).issubset(EFFECTIVE_AUTHORIZED_IMPORTS)
    assert "requests" in EFFECTIVE_AUTHORIZED_IMPORTS
    assert "playwright.sync_api" in EFFECTIVE_AUTHORIZED_IMPORTS


def test_effective_eda_imports_include_smolagents_safe_defaults() -> None:
    assert "datetime" in DEFAULT_SAFE_IMPORTS
    assert set(DEFAULT_SAFE_IMPORTS).issubset(EFFECTIVE_EDA_NOTEBOOK_IMPORTS)
    assert "pandas" in EFFECTIVE_EDA_NOTEBOOK_IMPORTS
    assert "seaborn" in EFFECTIVE_EDA_NOTEBOOK_IMPORTS


def test_backend_registers_search_tools() -> None:
    backend = SmolagentsCollectionBackend(model=object())

    assert [tool.name for tool in backend.tools] == ["web_search", "github_code_search"]


def test_collection_backend_mounts_project_for_pdf_helper() -> None:
    backend = SmolagentsCollectionBackend(model=object())

    kwargs = backend._build_executor_kwargs({"allow_network": False})

    assert kwargs["container_run_kwargs"]["shm_size"] == "1g"
    assert kwargs["container_run_kwargs"]["volumes"][str(PROJECT_ROOT)]["bind"] == "/workspace"
    assert kwargs["container_run_kwargs"]["working_dir"] == "/workspace"


def test_collection_backend_allows_overriding_shm_size() -> None:
    backend = SmolagentsCollectionBackend(model=object(), llm_config={"sandbox": {"shm_size": "2g"}})

    kwargs = backend._build_executor_kwargs({"allow_network": False})

    assert kwargs["container_run_kwargs"]["shm_size"] == "2g"


def test_collect_passes_search_tools_to_code_agent(monkeypatch) -> None:
    captured = {}

    class DummyExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["tools"] = kwargs["tools"]
            captured["instructions"] = kwargs["instructions"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run(self, task, max_steps: int, return_full_result: bool):
            return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(smolagents_backend_module, "PrebakedDockerExecutor", DummyExecutor)
    monkeypatch.setattr(smolagents_backend_module, "CodeAgent", DummyAgent)

    backend = SmolagentsCollectionBackend(model=object())
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert [tool.name for tool in captured["tools"]] == ["web_search", "github_code_search"]
    assert "agentic collection layer" in captured["instructions"]


def test_collect_defaults_to_twenty_steps_per_attempt(monkeypatch) -> None:
    captured = {}

    class DummyExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["agent_max_steps"] = kwargs["max_steps"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run(self, task, max_steps: int, return_full_result: bool):
            captured["run_max_steps"] = max_steps
            return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(smolagents_backend_module, "PrebakedDockerExecutor", DummyExecutor)
    monkeypatch.setattr(smolagents_backend_module, "CodeAgent", DummyAgent)

    backend = SmolagentsCollectionBackend(model=object())
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert captured["agent_max_steps"] == 20
    assert captured["run_max_steps"] == 20


def test_collect_uses_agent_config_defaults(monkeypatch) -> None:
    captured = {}

    class DummyExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyAgent:
        def __init__(self, *args, **kwargs) -> None:
            captured["agent_max_steps"] = kwargs["max_steps"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run(self, task, max_steps: int, return_full_result: bool):
            captured["run_max_steps"] = max_steps
            return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(smolagents_backend_module, "PrebakedDockerExecutor", DummyExecutor)
    monkeypatch.setattr(smolagents_backend_module, "CodeAgent", DummyAgent)

    backend = SmolagentsCollectionBackend(model=object(), agent_config={"max_steps": 7})
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert captured["agent_max_steps"] == 7
    assert captured["run_max_steps"] == 7


def test_build_container_env_includes_github_token_from_flat_config() -> None:
    backend = SmolagentsCollectionBackend(model=object(), llm_config={"github_token": "test-token"})

    environment = backend._build_container_env()

    assert environment["GITHUB_TOKEN"] == "test-token"


def test_build_container_env_includes_github_token_from_nested_config() -> None:
    backend = SmolagentsCollectionBackend(
        model=object(),
        llm_config={"github": {"token": "nested-token"}},
    )

    environment = backend._build_container_env()

    assert environment["GITHUB_TOKEN"] == "nested-token"


def test_notebook_backend_uses_restricted_import_set() -> None:
    assert "IPython" in EDA_NOTEBOOK_IMPORTS
    assert "pandas" in EDA_NOTEBOOK_IMPORTS
    assert "seaborn" in EDA_NOTEBOOK_IMPORTS
    assert "requests" not in EDA_NOTEBOOK_IMPORTS


def test_notebook_backend_loads_skill_instructions() -> None:
    backend = SmolagentsNotebookBackend(model=object())

    assert "You generate a Jupyter notebook" in backend.instructions


def test_notebook_backend_loads_inspection_instructions() -> None:
    backend = SmolagentsNotebookBackend(model=object())

    assert "You inspect the unified dataset" in backend.inspection_instructions


def test_notebook_backend_uses_dedicated_sandbox_defaults() -> None:
    backend = SmolagentsNotebookBackend(model=object())

    kwargs = backend._build_executor_kwargs({"allow_network": False})

    assert kwargs["image_name"] == "data-agent-eda-sandbox"
    assert kwargs["port"] == 8890
    assert kwargs["container_run_kwargs"]["shm_size"] == "1g"
    assert kwargs["container_run_kwargs"]["volumes"][str(PROJECT_ROOT)]["bind"] == "/workspace"


def test_notebook_backend_picks_free_port_when_default_is_busy(monkeypatch) -> None:
    backend = SmolagentsNotebookBackend(model=object())

    monkeypatch.setattr(smolagents_backend_module, "_is_local_port_free", lambda host, port: False)
    monkeypatch.setattr(smolagents_backend_module, "_find_free_local_port", lambda: 19090)

    assert backend._resolve_executor_port(8890) == 19090
