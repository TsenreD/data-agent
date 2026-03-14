from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class AgentResult:
    dataframe: pd.DataFrame | None = None
    dataframe_path: Path | None = None
    dataframe_schema: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    @abstractmethod
    def execute(self, payload: Any | None = None) -> AgentResult:
        raise NotImplementedError
