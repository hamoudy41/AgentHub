"""Embedding service for RAG."""

from __future__ import annotations

import hashlib
from typing import List

from ..core.config import get_settings


class EmbeddingService:
    """Service for embedding text. Uses mock for tests, extensible for OpenAI/sentence-transformers."""

    async def embed(self, text: str) -> List[float]:
        """Embed a single text and return a vector."""
        settings = get_settings()
        dim = settings.embedding_dimension
        if settings.embedding_model == "mock":
            return self._mock_embed(text, dim)
        # Future: OpenAI, sentence-transformers, etc.
        return self._mock_embed(text, dim)

    def _mock_embed(self, text: str, dim: int) -> List[float]:
        """Deterministic mock embedding based on text hash."""
        h = hashlib.sha256(text.encode()).digest()
        return [float((h[i % len(h)] - 128) / 128.0) for i in range(dim)]


embedding_service = EmbeddingService()
