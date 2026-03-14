import json
import re
from typing import Any, Mapping, Sequence

import requests

from .base import BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.trust_env = False

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
        response = self._session.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()

        content = self._extract_content(response.json())
        if json_schema:
            return self._parse_structured_content(content, json_schema)
        return content

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
        candidates = [content, self._strip_markdown_fence(content)]
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
