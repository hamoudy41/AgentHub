"""Sentence Transformers embedding provider implementation."""

from __future__ import annotations

from typing import Optional

from app.core.config import Settings
from app.core.logging import get_logger

from ..embedding import EmbeddingError, EmbeddingProvider

logger = get_logger(__name__)


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider (CPU or GPU accelerated).

    Uses HuggingFace sentence-transformers library for local embeddings.
    Can run on CPU (slower but portable) or GPU (faster).

    **Note**: This provider requires sentence-transformers package to be installed.

    Args:
        model_name: Model identifier from HuggingFace (e.g., 'all-MiniLM-L6-v2')
        settings: Optional settings for configuration
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize Sentence Transformers embedding provider.

        Args:
            model_name: HuggingFace model identifier
            settings: Optional application settings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._settings = settings
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "embedding.sentence_transformers_initialized",
            model=model_name,
            dimension=self._dimension,
        )

    async def embed(self, text: str) -> list[float]:
        """Embed text using Sentence Transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding fails
        """
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as exc:
            logger.error("embedding.sentence_transformers_error", error=str(exc))
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Dimension of embeddings
        """
        return self._dimension

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name (model identifier)
        """
        return self._model_name
