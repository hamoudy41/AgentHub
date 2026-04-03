"""Search provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.core.errors import AppError


class SearchError(AppError):
    """Error in search operation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "SEARCH_ERROR", 503)


@dataclass(frozen=True)
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    metadata: Optional[dict] = None


class SearchProvider(ABC):
    """Abstract base for search providers (DuckDuckGo, Tavily, etc.).

    Implementations of this interface enable pluggable search backends
    for tools like web search within agent workflows.
    """

    @abstractmethod
    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Search and return top results.

        Args:
            query: Search query string
            count: Number of results to return (default: 5)

        Returns:
            List of SearchResult objects ranked by relevance

        Raises:
            SearchError: If search operation fails
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass
