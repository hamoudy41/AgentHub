"""Search tool for the agent (mock implementation)."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def search_tool(query: str) -> str:
    """Search for information on the web. Use when you need current or external information."""
    # Mock implementation: return a placeholder. In production, integrate with Tavily, Serper, etc.
    return f"[Mock search] No results for: {query}. Configure a real search provider (Tavily, Serper) for production."
