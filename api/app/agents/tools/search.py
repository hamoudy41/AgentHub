"""Search tool for the agent - DuckDuckGo (default) or Tavily."""

from __future__ import annotations

from contextlib import nullcontext

from langchain_core.tools import tool

_MAX_TITLE_CHARS = 160
_MAX_SNIPPET_CHARS = 400
_MAX_OUTPUT_CHARS = 6000
_MAX_RESULTS_LIMIT = 10


class SearchToolError(RuntimeError):
    """Raised when a web-search provider fails."""


class SearchToolNoResults(LookupError):
    """Raised when a web-search provider returns no results."""


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_results(
    *,
    query: str,
    results: list[dict],
    title_key: str,
    snippet_key: str,
    url_key: str,
) -> str:
    if not results:
        raise SearchToolNoResults(query)

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = _truncate(str(r.get(title_key, "") or ""), _MAX_TITLE_CHARS)
        snippet = _truncate(str(r.get(snippet_key, "") or ""), _MAX_SNIPPET_CHARS)
        url = str(r.get(url_key, "") or "").strip()

        if not (title or snippet or url):
            continue

        block = [f"[{i}] {title}".rstrip()]
        if snippet:
            block.append(snippet)
        if url:
            block.append(f"Source: {url}")
        parts.append("\n".join(block).strip())

    if not parts:
        raise SearchToolNoResults(query)

    out = "\n\n".join(parts)
    if len(out) > _MAX_OUTPUT_CHARS:
        out = out[:_MAX_OUTPUT_CHARS].rstrip() + "\n\n(Results truncated.)"
    return out


def _search_duckduckgo(query: str, max_results: int = 5, region: str = "us-en") -> str:
    """Perform web search via DuckDuckGo (free, no API key)."""
    try:
        from duckduckgo_search import DDGS

        ddgs = DDGS()
        ctx = ddgs if hasattr(ddgs, "__enter__") else nullcontext(ddgs)
        with ctx as d:
            raw = d.text(query, region=region, max_results=max_results)
            results = list(raw or [])
        return _format_results(
            query=query,
            results=results,
            title_key="title",
            snippet_key="body",
            url_key="href",
        )
    except SearchToolNoResults:
        raise
    except Exception as e:
        raise SearchToolError(str(e)) from e


def _search_tavily(query: str, max_results: int = 5) -> str:
    """Perform web search via Tavily (better quality, requires TAVILY_API_KEY)."""
    import os

    from app.core.config import get_settings

    settings = get_settings()
    if settings.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results, topic="general")
        result = tool.invoke({"query": query})
        results = []
        if isinstance(result, dict):
            results = result.get("results") or []
        return _format_results(
            query=query,
            results=results,
            title_key="title",
            snippet_key="content",
            url_key="url",
        )
    except SearchToolNoResults:
        raise
    except Exception as e:
        raise SearchToolError(str(e)) from e


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using the configured provider."""
    from app.core.config import get_settings

    settings = get_settings()
    try:
        max_results = max(1, min(int(max_results), _MAX_RESULTS_LIMIT))
    except Exception:
        max_results = 5
    region = settings.search_region or "us-en"
    if settings.search_provider == "tavily" and settings.tavily_api_key:
        return _search_tavily(query, max_results)
    return _search_duckduckgo(query, max_results, region=region)


@tool
def search_tool(query: str, max_results: int = 5) -> str:
    """Web search. Prefer specific, distinctive terms; avoid generic words."""
    try:
        return search_web(query, max_results=max_results)
    except SearchToolNoResults:
        return f"No web results found for: {query}"
    except SearchToolError as e:
        return f"Search failed: {e}. Try rephrasing your question."
