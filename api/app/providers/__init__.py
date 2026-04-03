"""Provider integrations: abstract interfaces and implementations for external services."""

# Abstract interfaces
from .embedding import EmbeddingError, EmbeddingProvider
from .llm import LLMError, LLMNotConfiguredError, LLMProvider, LLMResult
from .search import SearchError, SearchProvider, SearchResult

# LLM implementations
from .llm_impl.ollama import OllamaProvider
from .llm_impl.openai import OpenAICompatibleProvider

# Embedding implementations
from .embedding_impl.mock import MockEmbeddingProvider
from .embedding_impl.openai import OpenAIEmbeddingProvider
from .embedding_impl.sentence_transformers import SentenceTransformersEmbeddingProvider

# Search implementations
from .search_impl.duckduckgo import DuckDuckGoSearchProvider
from .search_impl.mock import MockSearchProvider
from .search_impl.tavily import TavilySearchProvider

# Registry/factory
from .registry import ProviderRegistry

__all__ = [
    # Abstract interfaces
    "LLMProvider",
    "LLMResult",
    "LLMError",
    "LLMNotConfiguredError",
    "EmbeddingProvider",
    "EmbeddingError",
    "SearchProvider",
    "SearchResult",
    "SearchError",
    # LLM implementations
    "OllamaProvider",
    "OpenAICompatibleProvider",
    # Embedding implementations
    "MockEmbeddingProvider",
    "SentenceTransformersEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    # Search implementations
    "MockSearchProvider",
    "DuckDuckGoSearchProvider",
    "TavilySearchProvider",
    # Registry/factory
    "ProviderRegistry",
]
