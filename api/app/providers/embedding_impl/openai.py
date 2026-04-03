"""OpenAI embedding provider implementation."""

from __future__ import annotations

from typing import Optional

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

from ..embedding import EmbeddingError, EmbeddingProvider

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text embedding provider.

    Uses OpenAI's embedding API for generating dense vectors.
    Requires valid OpenAI API key in settings.

    Args:
        settings: Application settings with llm_api_key and base URL
        model: Model identifier (default: 'text-embedding-3-small')
    """

    def __init__(
        self,
        settings: Settings,
        model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            settings: Application settings
            model: Embedding model identifier
        """
        if not settings.llm_api_key:
            raise EmbeddingError("OpenAI API key required (llm_api_key not set)")

        self._settings = settings
        self._model = model
        self._dimension: Optional[int] = None  # Lazy loaded
        logger.info("embedding.openai_initialized", model=model)

    async def embed(self, text: str) -> list[float]:
        """Embed text using OpenAI embedding API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If API request fails
        """
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": text,
            "model": self._model,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    error_msg = response.text[:500]
                    raise EmbeddingError(f"OpenAI API error {response.status_code}: {error_msg}")
                data = response.json()
        except httpx.RequestError as exc:
            logger.error("embedding.openai_request_error", error=str(exc))
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc
        except ValueError as exc:
            logger.error("embedding.openai_invalid_json", error=str(exc))
            raise EmbeddingError("OpenAI returned invalid JSON") from exc

        embeddings = data.get("data") or []
        if not embeddings:
            raise EmbeddingError("OpenAI returned no embeddings")

        embedding = embeddings[0].get("embedding") or []
        if not isinstance(embedding, list):
            raise EmbeddingError("Invalid embedding format from OpenAI")

        if not self._dimension:
            self._dimension = len(embedding)

        return embedding

    def get_dimension(self) -> int:
        """Get embedding vector dimension.

        Returns:
            Dimension of embeddings

        Raises:
            EmbeddingError: If dimension unknown (call embed() first)
        """
        if self._dimension is None:
            raise EmbeddingError("Dimension unknown. Call embed() first to initialize.")
        return self._dimension

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name (model identifier)
        """
        return self._model
