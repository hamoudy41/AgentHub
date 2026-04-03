"""Provider registry and factory pattern for runtime provider selection."""

from __future__ import annotations

from typing import Optional

from app.core.config import Settings
from app.core.logging import get_logger

from .embedding import EmbeddingError, EmbeddingProvider
from .embedding_impl.mock import MockEmbeddingProvider
from .embedding_impl.openai import OpenAIEmbeddingProvider
from .embedding_impl.sentence_transformers import SentenceTransformersEmbeddingProvider
from .llm import LLMNotConfiguredError, LLMProvider
from .llm_impl.ollama import OllamaProvider
from .llm_impl.openai import OpenAICompatibleProvider
from .search import SearchError, SearchProvider
from .search_impl.duckduckgo import DuckDuckGoSearchProvider
from .search_impl.mock import MockSearchProvider
from .search_impl.tavily import TavilySearchProvider

logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for managing and creating provider instances.

    Provides a centralized factory for instantiating LLM, embedding, and search
    providers based on configuration. Enables runtime switching between implementations.

    Example:
        registry = ProviderRegistry(settings)
        llm_provider = registry.get_llm_provider()
        embedding_provider = registry.get_embedding_provider()
        search_provider = registry.get_search_provider()
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize provider registry.

        Args:
            settings: Application settings with provider configuration
        """
        self._settings = settings
        self._llm_cache: Optional[LLMProvider] = None
        self._embedding_cache: Optional[EmbeddingProvider] = None
        self._search_cache: Optional[SearchProvider] = None

    def get_llm_provider(self, force_new: bool = False) -> LLMProvider:
        """Get configured LLM provider (cached by default).

        Args:
            force_new: If True, create new instance ignoring cache

        Returns:
            Configured LLM provider instance

        Raises:
            LLMNotConfiguredError: If provider type not supported
        """
        if not force_new and self._llm_cache:
            return self._llm_cache

        provider_type = (self._settings.llm_provider or "ollama").lower().strip()
        logger.info("llm.provider_selection", provider_type=provider_type)

        if provider_type == "ollama":
            provider = OllamaProvider(self._settings)
        elif provider_type in ("openai", "openai-compatible", "openai_compatible", "azure"):
            provider = OpenAICompatibleProvider(self._settings)
        else:
            raise LLMNotConfiguredError(
                f"Unknown LLM provider: {provider_type}. "
                f"Supported: ollama, openai, openai-compatible, openai_compatible, azure"
            )

        self._llm_cache = provider
        return provider

    def get_embedding_provider(self, force_new: bool = False) -> EmbeddingProvider:
        """Get configured embedding provider (cached by default).

        Args:
            force_new: If True, create new instance ignoring cache

        Returns:
            Configured embedding provider instance

        Raises:
            EmbeddingError: If provider type not supported
        """
        if not force_new and self._embedding_cache:
            return self._embedding_cache

        provider_type = (self._settings.embedding_provider or "mock").lower().strip()
        logger.info("embedding.provider_selection", provider_type=provider_type)

        if provider_type == "mock":
            provider = MockEmbeddingProvider(dimension=384)
        elif provider_type == "sentence-transformers":
            model = self._settings.embedding_model or "all-MiniLM-L6-v2"
            provider = SentenceTransformersEmbeddingProvider(
                model_name=model, settings=self._settings
            )
        elif provider_type == "openai":
            provider = OpenAIEmbeddingProvider(
                self._settings,
                model=self._settings.embedding_model or "text-embedding-3-small",
            )
        else:
            raise EmbeddingError(
                f"Unknown embedding provider: {provider_type}. "
                f"Supported: mock, sentence-transformers, openai"
            )

        self._embedding_cache = provider
        return provider

    def get_search_provider(self, force_new: bool = False) -> SearchProvider:
        """Get configured search provider (cached by default).

        Args:
            force_new: If True, create new instance ignoring cache

        Returns:
            Configured search provider instance

        Raises:
            SearchError: If provider type not supported or not configured
        """
        if not force_new and self._search_cache:
            return self._search_cache

        provider_type = (self._settings.search_provider or "mock").lower().strip()
        logger.info("search.provider_selection", provider_type=provider_type)

        if provider_type == "mock":
            provider = MockSearchProvider()
        elif provider_type == "duckduckgo":
            provider = DuckDuckGoSearchProvider()
        elif provider_type == "tavily":
            api_key = self._settings.tavily_api_key
            if not api_key:
                raise SearchError("Tavily API key required (tavily_api_key not set in settings)")
            provider = TavilySearchProvider(api_key=api_key, settings=self._settings)
        else:
            raise SearchError(
                f"Unknown search provider: {provider_type}. Supported: mock, duckduckgo, tavily"
            )

        self._search_cache = provider
        return provider

    def reset_caches(self) -> None:
        """Clear all cached provider instances.

        Useful for testing or when configuration changes at runtime.
        """
        self._llm_cache = None
        self._embedding_cache = None
        self._search_cache = None
        logger.info("registry.caches_reset")
