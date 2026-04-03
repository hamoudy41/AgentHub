"""Mock embedding provider for testing."""

from __future__ import annotations

import hashlib

from app.core.logging import get_logger

from ..embedding import EmbeddingProvider

logger = get_logger(__name__)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing and development.

    Generates deterministic embeddings based on text hashing.
    Useful for testing without external dependencies.
    Not suitable for production use.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock embedding provider.

        Args:
            dimension: Size of embedding vectors (default 384)
        """
        self._dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of floats as mock embedding
        """
        # Hash text and convert to normalized vector
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Generate embedding by converting hash to floats
        embedding = []
        for i in range(self._dimension):
            byte_val = int(hash_hex[i * 2 : i * 2 + 2], 16) if i * 2 < len(hash_hex) else 0
            # Normalize to [-1, 1]
            normalized = (byte_val / 128.0) - 1.0
            embedding.append(float(normalized))

        logger.debug("embedding.mock_generated", text_length=len(text), dimension=len(embedding))
        return embedding

    def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Dimension of embeddings
        """
        return self._dimension

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name: "mock"
        """
        return "mock"
