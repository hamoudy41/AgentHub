"""DuckDuckGo search provider implementation."""

from __future__ import annotations


import httpx

from app.core.logging import get_logger

from ..search import SearchError, SearchProvider, SearchResult

logger = get_logger(__name__)

_DDG_API_URL = "https://api.duckduckgo.com/"
_DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


async def _fetch_ddg_payload(query: str) -> dict:
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1,
    }

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        response = await client.get(_DDG_API_URL, params=params, headers=_DDG_HEADERS)
        if response.status_code != 200:
            raise SearchError(f"DuckDuckGo returned {response.status_code}")
        return response.json()


def _collect_related_topics(items: list[dict], results: list[SearchResult], limit: int) -> None:
    for item in items:
        if len(results) >= limit:
            return
        if not isinstance(item, dict):
            continue

        nested = item.get("Topics")
        if isinstance(nested, list):
            _collect_related_topics(nested, results, limit)
            continue

        text = item.get("Text")
        first_url = item.get("FirstURL")
        if isinstance(text, str) and isinstance(first_url, str):
            results.append(
                SearchResult(
                    title=(item.get("Title") or text.split(" - ")[0]).strip(),
                    url=first_url,
                    snippet=text,
                    metadata={"source": "duckduckgo"},
                )
            )


def _extract_results(data: dict, limit: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    _collect_related_topics(data.get("RelatedTopics") or [], results, limit)
    return results[:limit]


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
        try:
            data = await _fetch_ddg_payload(query)
        except httpx.RequestError as exc:
            logger.error("search.duckduckgo_request_error", error=str(exc), query=query)
            raise SearchError(f"DuckDuckGo request failed: {exc}") from exc
        except ValueError as exc:
            logger.error("search.duckduckgo_invalid_json", error=str(exc))
            raise SearchError("DuckDuckGo returned invalid JSON") from exc
        results = _extract_results(data, limit)

        logger.info(
            "search.duckduckgo_results",
            query=query,
            num_results=len(results),
            requested=limit,
        )
        return results

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name: 'duckduckgo'
        """
        return "duckduckgo"
