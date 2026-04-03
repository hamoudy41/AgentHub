"""Tavily search provider implementation."""

from __future__ import annotations

from typing import Optional

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

from ..search import SearchError, SearchProvider, SearchResult

logger = get_logger(__name__)


class TavilySearchProvider(SearchProvider):
    """Tavily AI search provider (paid service, premium results).

    Provides high-quality search results with semantic understanding.
    Requires Tavily API key from https://tavily.com.

    Args:
        api_key: Tavily API key
        settings: Optional application settings
    """

    def __init__(
        self,
        api_key: str,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key
            settings: Optional application settings
        """
        if not api_key:
            raise SearchError("Tavily API key required")

        self._api_key = api_key
        self._settings = settings
        logger.info("search.tavily_initialized")

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search using Tavily AI.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If search request fails
        """
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": limit,
            "include_answer": True,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    error_msg = response.text[:500]
                    raise SearchError(f"Tavily returned {response.status_code}: {error_msg}")
                data = response.json()
        except httpx.RequestError as exc:
            logger.error("search.tavily_request_error", error=str(exc), query=query)
            raise SearchError(f"Tavily request failed: {exc}") from exc
        except ValueError as exc:
            logger.error("search.tavily_invalid_json", error=str(exc))
            raise SearchError("Tavily returned invalid JSON") from exc

        results = []
        hits = data.get("results") or []

        for item in hits[:limit]:
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                metadata={
                    "source": "tavily",
                    "score": item.get("score"),
                },
            )
            results.append(result)

        logger.info(
            "search.tavily_results",
            query=query,
            num_results=len(results),
            requested=limit,
        )
        return results[:limit]

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name: 'tavily'
        """
        return "tavily"
