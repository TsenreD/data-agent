import json
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

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": prepared_messages,
            "stream": False,
        }
        if json_schema:
            payload["format"] = "json"

        response = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        content = response.json()["message"]["content"]
        if json_schema:
            return json.loads(content)
        return content
