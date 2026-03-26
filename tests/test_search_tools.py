import requests

from agents.tools import DuckDuckGoWebSearchTool, GitHubCodeSearchTool, build_search_tools


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self):
        return self._payload


def test_build_search_tools_returns_both_tools() -> None:
    tools = build_search_tools()

    assert [tool.name for tool in tools] == ["web_search", "github_code_search"]


def test_search_tools_can_be_serialized_for_remote_execution() -> None:
    tools = build_search_tools()

    for tool in tools:
        tool_dict = tool.to_dict()
        assert "code" in tool_dict
        assert "requirements" in tool_dict


def test_web_search_tool_formats_duckduckgo_results(monkeypatch) -> None:
    class FakeDDGS:
        def text(self, query: str, max_results: int):
            assert query == "httpx ReadTimeout"
            assert max_results == 2
            return [
                {
                    "title": "HTTPX timeouts",
                    "href": "https://www.python-httpx.org/advanced/timeouts/",
                    "body": "Fine tune connect, read, write, and pool timeouts.",
                }
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)

    result = DuckDuckGoWebSearchTool().forward("httpx ReadTimeout", 2)

    assert result["ok"] is True
    assert result["results"] == [
        {
            "title": "HTTPX timeouts",
            "url": "https://www.python-httpx.org/advanced/timeouts/",
            "snippet": "Fine tune connect, read, write, and pool timeouts.",
        }
    ]


def test_github_code_search_tool_formats_api_results(monkeypatch) -> None:
    def fake_get(url: str, *, params, headers, timeout: int):
        assert url == "https://api.github.com/search/code"
        assert params == {"q": "TimeoutException language:python", "per_page": 2}
        assert headers["Accept"] == "application/vnd.github.text-match+json"
        assert timeout == 20
        return _FakeResponse(
            {
                "items": [
                    {
                        "repository": {"full_name": "encode/httpx"},
                        "path": "httpx/_exceptions.py",
                        "html_url": "https://github.com/encode/httpx/blob/master/httpx/_exceptions.py",
                        "score": 12.5,
                        "text_matches": [{"fragment": "class TimeoutException(TransportError):"}],
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)

    result = GitHubCodeSearchTool().forward("TimeoutException language:python", 2)

    assert result["ok"] is True
    assert result["results"] == [
        {
            "repository": "encode/httpx",
            "path": "httpx/_exceptions.py",
            "url": "https://github.com/encode/httpx/blob/master/httpx/_exceptions.py",
            "score": 12.5,
            "fragments": ["class TimeoutException(TransportError):"],
        }
    ]


def test_github_code_search_tool_returns_api_errors(monkeypatch) -> None:
    def fake_get(url: str, *, params, headers, timeout: int):
        return _FakeResponse({"message": "Requires authentication"}, status_code=403)

    monkeypatch.setattr(requests, "get", fake_get)

    result = GitHubCodeSearchTool().forward("repo:private/project bug", 3)

    assert result["ok"] is False
    assert result["error"] == "HTTPError: Requires authentication"
