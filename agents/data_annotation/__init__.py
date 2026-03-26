from typing import TYPE_CHECKING


__all__ = ["DataAnnotationAgent"]


if TYPE_CHECKING:
    from .data_annotation_agent import DataAnnotationAgent


def __getattr__(name: str):
    if name == "DataAnnotationAgent":
        from .data_annotation_agent import DataAnnotationAgent

        return DataAnnotationAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
