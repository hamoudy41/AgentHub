"""Search tool for the agent - DuckDuckGo (default) or Tavily."""

from __future__ import annotations

from langchain_core.tools import tool


def _search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Perform web search via DuckDuckGo (free, no API key)."""
    try:
        from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No web results found for: {query}"
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            if title or body:
                parts.append(f"[{i}] {title}\n{body}\nSource: {href}")
        return "\n\n".join(parts) if parts else f"No web results found for: {query}"
    except Exception as e:
        return f"Search failed: {e}. Try rephrasing your question."


def _search_tavily(query: str, max_results: int = 5) -> str:
    """Perform web search via Tavily (better quality, requires TAVILY_API_KEY)."""
    import os

    from app.core.config import get_settings

    settings = get_settings()
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results, topic="general")
        result = tool.invoke({"query": query})
        if not result or not isinstance(result, dict):
            return f"No web results found for: {query}"
        results = result.get("results") or []
        if not results:
            return f"No web results found for: {query}"
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            if title or content:
                parts.append(f"[{i}] {title}\n{content}\nSource: {url}")
        return "\n\n".join(parts) if parts else f"No web results found for: {query}"
    except Exception as e:
        return f"Search failed: {e}. Try rephrasing your question."


def _search_web(query: str, max_results: int = 5) -> str:
    """Perform web search using configured provider."""
    from app.core.config import get_settings

    settings = get_settings()
    if settings.search_provider == "tavily" and settings.tavily_api_key:
        return _search_tavily(query, max_results)
    return _search_duckduckgo(query, max_results)


@tool
def search_tool(query: str) -> str:
    """Search for information on the web. Use the most specific, distinctive terms from the user's question. Avoid single generic words."""
    return _search_web(query)
