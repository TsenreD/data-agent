from typing import TYPE_CHECKING


__all__ = ["DataCollectionAgent"]


if TYPE_CHECKING:
    from .data_collection_agent import DataCollectionAgent


def __getattr__(name: str):
    if name == "DataCollectionAgent":
        from .data_collection_agent import DataCollectionAgent

        return DataCollectionAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
