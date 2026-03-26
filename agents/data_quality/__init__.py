from typing import TYPE_CHECKING


__all__ = ["DataQualityAgent"]


if TYPE_CHECKING:
    from .data_quality_agent import DataQualityAgent


def __getattr__(name: str):
    if name == "DataQualityAgent":
        from .data_quality_agent import DataQualityAgent

        return DataQualityAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
