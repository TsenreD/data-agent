from smolagents import Tool


class DuckDuckGoWebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the public web with DuckDuckGo for documentation, error messages, library usage, or selectors. "
        "Use this when generated code fails and you need external references. Returns a structured object with "
        "search results containing titles, URLs, and snippets."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query. Include the exact error message, package name, API, or selector you need.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Keep this small and focused.",
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(self) -> None:
        super().__init__()
        self.output_schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "ok": {"type": "boolean"},
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                },
                "error": {"type": ["string", "null"]},
            },
        }

    def _clip(self, value, limit: int = 400) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _response(
        self,
        query: str,
        ok: bool,
        results: list[dict] | None = None,
        error: str | None = None,
    ) -> dict:
        return {
            "tool": self.name,
            "ok": ok,
            "query": query,
            "results": results or [],
            "error": error,
        }

    def forward(self, query: str, max_results: int = 5) -> dict:
        from ddgs import DDGS

        limit = max(1, min(int(max_results), 10))
        try:
            raw_results = DDGS().text(query, max_results=limit)
            results = [
                {
                    "title": self._clip(item.get("title"), 200) or "",
                    "url": item.get("href") or item.get("url") or "",
                    "snippet": self._clip(item.get("body") or item.get("snippet"), 400),
                }
                for item in raw_results[:limit]
            ]
            return self._response(query=query, ok=True, results=results)
        except Exception as error:
            return self._response(query=query, ok=False, error=f"{type(error).__name__}: {error}")


class GitHubCodeSearchTool(Tool):
    name = "github_code_search"
    description = (
        "Search public GitHub code to find examples, fixes, and prior art for a failing implementation. "
        "Use standard GitHub code search qualifiers directly in the query, for example "
        "'httpx ReadTimeout language:python' or 'repo:encode/httpx TimeoutException'. "
        "Returns a structured object with repository, path, URL, and matched text fragments when GitHub provides them."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "GitHub code search query. You can include qualifiers like repo:, path:, org:, and language:.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of code matches to return. Keep this small and focused.",
            "nullable": True,
        },
    }
    output_type = "object"

    def __init__(self) -> None:
        super().__init__()
        self.output_schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "ok": {"type": "boolean"},
                "query": {"type": "string"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "repository": {"type": "string"},
                            "path": {"type": "string"},
                            "url": {"type": "string"},
                            "score": {"type": "number"},
                            "fragments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "error": {"type": ["string", "null"]},
            },
        }

    def _clip(self, value, limit: int = 400) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _response(
        self,
        query: str,
        ok: bool,
        results: list[dict] | None = None,
        error: str | None = None,
    ) -> dict:
        return {
            "tool": self.name,
            "ok": ok,
            "query": query,
            "results": results or [],
            "error": error,
        }

    def forward(self, query: str, max_results: int = 5) -> dict:
        import os

        import requests

        limit = max(1, min(int(max_results), 10))
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "User-Agent": "data-agent-smolagents",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = requests.get(
                "https://api.github.com/search/code",
                params={"q": query, "per_page": limit},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            results = []
            for item in payload.get("items", [])[:limit]:
                fragments = [
                    self._clip(match.get("fragment"), 400) or ""
                    for match in item.get("text_matches", [])
                    if match.get("fragment")
                ]
                results.append(
                    {
                        "repository": item.get("repository", {}).get("full_name") or "",
                        "path": item.get("path") or "",
                        "url": item.get("html_url") or "",
                        "score": float(item.get("score", 0.0)),
                        "fragments": fragments,
                    }
                )
            return self._response(query=query, ok=True, results=results)
        except requests.HTTPError as error:
            message = None
            try:
                payload = error.response.json()
                message = payload.get("message")
            except Exception:
                message = None
            detail = message or str(error)
            return self._response(query=query, ok=False, error=f"HTTPError: {detail}")
        except Exception as error:
            return self._response(query=query, ok=False, error=f"{type(error).__name__}: {error}")


def build_search_tools() -> list[Tool]:
    return [DuckDuckGoWebSearchTool(), GitHubCodeSearchTool()]
