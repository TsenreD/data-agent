import json
import re
import time
from typing import Any, Mapping, Sequence

import requests

from .base import BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int | None = None,
        headers: Mapping[str, str] | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_max_seconds = max(self.retry_backoff_seconds, float(retry_backoff_max_seconds))
        self._session = requests.Session()
        self._session.trust_env = False
        self._headers = dict(headers or {})
        if api_key and "Authorization" not in self._headers:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        prepared_messages = list(messages)
        if json_schema:
            prepared_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON that matches this schema exactly:\n"
                        f"{json.dumps(json_schema)}"
                    ),
                },
                *prepared_messages,
            ]

        endpoint, payload = self._build_request(prepared_messages, json_schema)
        response = self._post_with_retries(endpoint, payload)

        content = self._extract_content(response.json())
        if json_schema:
            return self._parse_structured_content(content, json_schema)
        return content

    def _post_with_retries(self, endpoint: str, payload: Mapping[str, Any]) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    endpoint,
                    json=payload,
                    headers=self._headers or None,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < self.max_retries:
                    time.sleep(self._retry_delay(attempt, response))
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as error:
                last_error = error
                if attempt + 1 >= self.max_retries or not self._is_retryable_exception(error):
                    raise
                time.sleep(self._retry_delay(attempt, getattr(error, "response", None)))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Model request failed without raising an exception.")

    def _is_retryable_exception(self, error: requests.exceptions.RequestException) -> bool:
        if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        response = getattr(error, "response", None)
        if response is None:
            return False
        return response.status_code in {429, 500, 502, 503, 504}

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(self.retry_backoff_max_seconds, max(0.0, float(retry_after)))
                except ValueError:
                    pass
        delay = self.retry_backoff_seconds * (2**attempt)
        return min(self.retry_backoff_max_seconds, delay)

    def _build_request(
        self,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if self.base_url.endswith("/v1/chat/completions"):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": list(messages),
                "stream": False,
            }
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": json_schema,
                    },
                }
            return self.base_url, payload

        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["options"] = {"num_predict": self.max_tokens}
        if json_schema:
            payload["format"] = "json"
        return f"{self.base_url}/api/chat", payload

    def _extract_content(self, payload: Mapping[str, Any]) -> str:
        if "message" in payload:
            return str(payload["message"]["content"])
        if "choices" in payload:
            return str(payload["choices"][0]["message"]["content"])
        raise ValueError("Unsupported model response payload.")

    def _parse_structured_content(
        self,
        content: str,
        json_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        extracted_json = self._extract_json_block(content)
        candidates = [content, self._strip_markdown_fence(content), extracted_json]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])
        if len(required) == 1:
            field_name = str(required[0])
            field_schema = properties.get(field_name, {})
            if field_schema.get("type") == "string":
                extracted = self._extract_code_block(content)
                if extracted:
                    return {field_name: extracted}
                stripped = content.strip()
                if stripped:
                    return {field_name: stripped}

        raise json.JSONDecodeError("Expecting value", content, 0)

    def _extract_json_block(self, content: str) -> str | None:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = content.find(opener)
            while start != -1:
                candidate = self._balanced_json_slice(content, start, opener, closer)
                if candidate:
                    return candidate
                start = content.find(opener, start + 1)
        return None

    def _balanced_json_slice(
        self,
        content: str,
        start: int,
        opener: str,
        closer: str,
    ) -> str | None:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        return None

    def _strip_markdown_fence(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def _extract_code_block(self, content: str) -> str | None:
        match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
