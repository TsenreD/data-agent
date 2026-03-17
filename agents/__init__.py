from typing import TYPE_CHECKING

from .base import AgentResult, BaseAgent

__all__ = ["AgentResult", "BaseAgent", "DataCollectionAgent", "PipelineRunner"]


if TYPE_CHECKING:
    from .data_collection.data_collection_agent import DataCollectionAgent
    from .pipeline import PipelineRunner


def __getattr__(name: str):
    if name == "DataCollectionAgent":
        from .data_collection.data_collection_agent import DataCollectionAgent

        return DataCollectionAgent
    if name == "PipelineRunner":
        from .pipeline import PipelineRunner

        return PipelineRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
