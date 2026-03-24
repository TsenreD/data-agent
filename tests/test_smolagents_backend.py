from agents.data_collection.smolagents_backend import (
    AUTHORIZED_IMPORTS,
    DEFAULT_SAFE_IMPORTS,
    EDA_NOTEBOOK_IMPORTS,
    EFFECTIVE_AUTHORIZED_IMPORTS,
    EFFECTIVE_EDA_NOTEBOOK_IMPORTS,
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
    backend = SmolagentsCollectionBackend(model=object(), agent_config={"execution_timeout_seconds": 0})
    records, notes = backend._parse_output('{"records":[{"text":"hello","title":"greeting"}],"notes":["ok"]}')

    assert records == [{"text": "hello", "title": "greeting"}]
    assert notes == ["ok"]


def test_parse_output_accepts_fenced_json() -> None:
    backend = SmolagentsCollectionBackend(model=object(), agent_config={"execution_timeout_seconds": 0})
    records, notes = backend._parse_output('```json\n{"records":[{"text":"hello"}]}\n```')

    assert records == [{"text": "hello"}]
    assert notes == []


def test_parse_output_accepts_python_literal_dict_string() -> None:
    backend = SmolagentsCollectionBackend(model=object(), agent_config={"execution_timeout_seconds": 0})
    records, notes = backend._parse_output("{'records': [{'text': 'hello'}], 'notes': ['ok']}")

    assert records == [{"text": "hello"}]
    assert notes == ["ok"]


def test_build_task_requires_github_search_before_writing_code() -> None:
    backend = SmolagentsCollectionBackend(model=object(), agent_config={"execution_timeout_seconds": 0})

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
    assert "Do not use search-result snippets as `text`." in task
    assert "You have at most 9 agent steps in this attempt." in task
    assert "try the provided helper modules first" in task


def test_all_imports_are_authorized() -> None:
    assert "*" not in AUTHORIZED_IMPORTS
    assert "requests" in AUTHORIZED_IMPORTS
    assert "agents.data_collection.pdf_utils" in AUTHORIZED_IMPORTS
    assert "agents.data_collection.web_utils" in AUTHORIZED_IMPORTS
    assert "playwright.sync_api" in AUTHORIZED_IMPORTS


def test_effective_authorized_imports_include_smolagents_safe_defaults() -> None:
    assert "datetime" in DEFAULT_SAFE_IMPORTS
    assert set(DEFAULT_SAFE_IMPORTS).issubset(EFFECTIVE_AUTHORIZED_IMPORTS)
    assert "requests" in EFFECTIVE_AUTHORIZED_IMPORTS


def test_effective_eda_imports_include_smolagents_safe_defaults() -> None:
    assert "datetime" in DEFAULT_SAFE_IMPORTS
    assert set(DEFAULT_SAFE_IMPORTS).issubset(EFFECTIVE_EDA_NOTEBOOK_IMPORTS)
    assert "pandas" in EFFECTIVE_EDA_NOTEBOOK_IMPORTS


def test_backend_registers_search_tools() -> None:
    backend = SmolagentsCollectionBackend(model=object(), agent_config={"execution_timeout_seconds": 0})

    assert [tool.name for tool in backend.tools] == ["web_search", "github_code_search"]


def test_collect_passes_search_tools_to_code_agent(monkeypatch) -> None:
    captured = {}

    def fake_run_agent(self, **kwargs):
        captured["tools"] = kwargs["tools"]
        captured["instructions"] = kwargs["instructions"]
        return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(SmolagentsCollectionBackend, "_run_agent", fake_run_agent)

    backend = SmolagentsCollectionBackend(
        model=object(),
        agent_config={"execution_timeout_seconds": 0},
    )
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert [tool.name for tool in captured["tools"]] == ["web_search", "github_code_search"]
    assert "agentic collection layer" in captured["instructions"]


def test_collect_defaults_to_twenty_steps_per_attempt(monkeypatch) -> None:
    captured = {}

    def fake_run_agent(self, **kwargs):
        captured["run_max_steps"] = kwargs["max_steps"]
        return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(SmolagentsCollectionBackend, "_run_agent", fake_run_agent)

    backend = SmolagentsCollectionBackend(model=object())
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert captured["run_max_steps"] == 20


def test_collect_defaults_to_twenty_second_execution_timeout(monkeypatch) -> None:
    captured = {}

    def fake_run_agent(self, **kwargs):
        captured["execution_timeout_seconds"] = kwargs["execution_timeout_seconds"]
        return RunResult(output='{"records":[{"text":"hello"}]}', state="done", steps=[])

    monkeypatch.setattr(SmolagentsCollectionBackend, "_run_agent", fake_run_agent)

    backend = SmolagentsCollectionBackend(model=object())
    result = backend.collect({"type": "scrape", "url": "https://example.com"})

    assert result.success is True
    assert captured["execution_timeout_seconds"] == 20


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
