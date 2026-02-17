"""Document lookup tool - fetches document by ID from the database."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.tools import tool


def create_document_lookup_tool(
    tenant_id: str,
    get_document_fn: Callable[[str, str], Awaitable[dict[str, Any] | None]],
) -> Any:
    """Create a document_lookup tool bound to tenant and async document fetcher."""

    @tool
    async def document_lookup_tool(document_id: str) -> str:
        """Look up a document by ID. Use when the user asks about a specific document or references a document ID."""
        try:
            doc = await get_document_fn(document_id, tenant_id)
            if not doc:
                return f"Document '{document_id}' not found."
            return f"Title: {doc.get('title', 'Untitled')}\n\nContent:\n{doc.get('text', '')}"
        except Exception as e:
            return f"Error fetching document: {e}"

    return document_lookup_tool
