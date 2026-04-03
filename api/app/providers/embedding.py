"""Embedding provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.errors import AppError


class EmbeddingError(AppError):
    """Error in embedding operation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR", 503)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed text and return vector."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass
