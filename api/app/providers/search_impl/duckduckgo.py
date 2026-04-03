"""DuckDuckGo search provider implementation."""

from __future__ import annotations


import httpx

from app.core.logging import get_logger

from ..search import SearchError, SearchProvider, SearchResult

logger = get_logger(__name__)


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo search provider (free, no API key required).

    Uses DuckDuckGo's public search interface for web search without authentication.
    Results are limited in accuracy and freshness compared to paid services.
    """

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Search using DuckDuckGo.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If search request fails
        """
        url = "https://duckduckgo.com/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
        }
        params = {
            "q": query,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code != 200:
                    raise SearchError(f"DuckDuckGo returned {response.status_code}")
                data = response.json()
        except httpx.RequestError as exc:
            logger.error("search.duckduckgo_request_error", error=str(exc), query=query)
            raise SearchError(f"DuckDuckGo request failed: {exc}") from exc
        except ValueError as exc:
            logger.error("search.duckduckgo_invalid_json", error=str(exc))
            raise SearchError("DuckDuckGo returned invalid JSON") from exc

        results = []
        related = data.get("RelatedTopics") or []

        for item in related[:limit]:
            if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
                result = SearchResult(
                    title=item.get("Title") or item.get("Text", "").split(" - ")[0],
                    url=item["FirstURL"],
                    snippet=item.get("Text", ""),
                    metadata={"source": "duckduckgo"},
                )
                results.append(result)

        logger.info(
            "search.duckduckgo_results",
            query=query,
            num_results=len(results),
            requested=limit,
        )
        return results[:limit]

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name: 'duckduckgo'
        """
        return "duckduckgo"
