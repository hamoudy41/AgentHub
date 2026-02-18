"""Domain interfaces for embeddings and vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Embedding:
    """Vector embedding for a text."""

    vector: list[float]
    dimension: int
    model: str


class IEmbeddingService(ABC):
    """Interface for embedding services."""

    @abstractmethod
    async def embed(self, text: str) -> Embedding:
        """Generate an embedding for the given text."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        pass
