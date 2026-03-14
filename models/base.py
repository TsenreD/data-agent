from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class BaseModelAdapter(ABC):
    @abstractmethod
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        raise NotImplementedError
