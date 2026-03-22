from __future__ import annotations

import hashlib
from typing import List

from ..core.config import get_settings


class EmbeddingService:
    async def embed(self, text: str) -> List[float]:
        settings = get_settings()
        dim = settings.embedding_dimension
        if settings.embedding_model == "mock":
            return self._mock_embed(text, dim)
        return self._mock_embed(text, dim)

    def _mock_embed(self, text: str, dim: int) -> List[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [float((h[i % len(h)] - 128) / 128.0) for i in range(dim)]


embedding_service = EmbeddingService()
