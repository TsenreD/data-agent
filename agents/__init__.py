from typing import TYPE_CHECKING

from .base import AgentResult, BaseAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    "ActiveLearningAgent",
    "DataCollectionAgent",
    "DataAnnotationAgent",
    "DataQualityAgent",
    "PipelineRunner",
]


if TYPE_CHECKING:
    from .active_learning.active_learning_agent import ActiveLearningAgent
    from .data_annotation.data_annotation_agent import DataAnnotationAgent
    from .data_collection.data_collection_agent import DataCollectionAgent
    from .data_quality.data_quality_agent import DataQualityAgent
    from .pipeline import PipelineRunner


def __getattr__(name: str):
    if name == "ActiveLearningAgent":
        from .active_learning.active_learning_agent import ActiveLearningAgent

        return ActiveLearningAgent
    if name == "DataAnnotationAgent":
        from .data_annotation.data_annotation_agent import DataAnnotationAgent

        return DataAnnotationAgent
    if name == "DataCollectionAgent":
        from .data_collection.data_collection_agent import DataCollectionAgent

        return DataCollectionAgent
    if name == "DataQualityAgent":
        from .data_quality.data_quality_agent import DataQualityAgent

        return DataQualityAgent
    if name == "PipelineRunner":
        from .pipeline import PipelineRunner

        return PipelineRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
