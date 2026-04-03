"""Mock search provider for testing."""

from __future__ import annotations

from app.core.logging import get_logger

from ..search import SearchProvider, SearchResult

logger = get_logger(__name__)


class MockSearchProvider(SearchProvider):
    """Mock search provider for testing and development.

    Returns pre-defined search results based on keyword matching.
    Useful for testing without external dependencies.
    Not suitable for production use.
    """

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return mock search results matching query keywords.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of mock SearchResult objects
        """
        # Simple keyword-based mock results
        mock_results = {
            "python": [
                SearchResult(
                    title="Python Official Website",
                    url="https://www.python.org",
                    snippet="Python is a high-level programming language.",
                    metadata={"source": "mock"},
                ),
                SearchResult(
                    title="Python Documentation",
                    url="https://docs.python.org",
                    snippet="Official Python documentation and guides.",
                    metadata={"source": "mock"},
                ),
            ],
            "machine learning": [
                SearchResult(
                    title="Machine Learning Fundamentals",
                    url="https://example.com/ml-basics",
                    snippet="Learn the basics of machine learning algorithms.",
                    metadata={"source": "mock"},
                ),
            ],
        }

        # Find matching results based on query keywords
        results = []
        query_lower = query.lower()

        for keyword, keyword_results in mock_results.items():
            if keyword in query_lower:
                results.extend(keyword_results)

        # If no matches, return generic result
        if not results:
            results = [
                SearchResult(
                    title=f"Search results for: {query}",
                    url="https://example.com/search",
                    snippet=f"Mock search results for query: {query}",
                    metadata={"source": "mock"},
                ),
            ]

        logger.debug(
            "search.mock_results",
            query=query,
            num_results=len(results),
            requested=limit,
        )
        return results[:limit]

    def get_name(self) -> str:
        """Get provider name.

        Returns:
            Provider name: 'mock'
        """
        return "mock"
