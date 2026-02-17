"""Tests for agent tools (TDD)."""

from __future__ import annotations

import pytest

from app.agents.tools import calculator_tool, search_tool
from app.agents.tools.document_lookup import create_document_lookup_tool


def test_calculator_tool_simple_arithmetic():
    """Calculator evaluates 2+3*4 correctly."""
    result = calculator_tool.invoke({"expression": "2+3*4"})
    assert result == "14"


def test_calculator_tool_division():
    """Calculator evaluates division."""
    result = calculator_tool.invoke({"expression": "100/5"})
    assert result == "20.0"


def test_calculator_tool_power():
    """Calculator evaluates exponentiation."""
    result = calculator_tool.invoke({"expression": "2**10"})
    assert result == "1024"


def test_calculator_tool_invalid_expression():
    """Calculator returns error for invalid expression."""
    result = calculator_tool.invoke({"expression": "1 + 'x'"})
    assert "Error" in result


def test_calculator_tool_single_number():
    """Calculator handles single number (Constant node)."""
    result = calculator_tool.invoke({"expression": "42"})
    assert result == "42"


def test_calculator_tool_unary_minus():
    """Calculator handles unary minus (UnaryOp)."""
    result = calculator_tool.invoke({"expression": "-10"})
    assert result == "-10"


def test_calculator_tool_floor_division():
    """Calculator evaluates floor division."""
    result = calculator_tool.invoke({"expression": "10//3"})
    assert result == "3"


def test_calculator_tool_modulo():
    """Calculator evaluates modulo."""
    result = calculator_tool.invoke({"expression": "10 % 3"})
    assert result == "1"


def test_search_tool_returns_results():
    """Search tool returns web search results or error message."""
    result = search_tool.invoke({"query": "weather Paris"})
    assert isinstance(result, str)
    assert len(result) > 0
    # Either real results, "No web results", or "Search failed"
    assert (
        "Paris" in result
        or "No web results" in result
        or "Search failed" in result
        or "Source:" in result
    )


def test_search_tool_uses_tavily_when_configured():
    """Search tool uses Tavily when SEARCH_PROVIDER=tavily and TAVILY_API_KEY set."""
    from unittest.mock import patch

    mock_tavily_result = {
        "results": [
            {"title": "Test Result", "content": "Test content here.", "url": "https://example.com"},
        ],
    }

    with (
        patch("app.core.config.get_settings") as mock_settings,
        patch("langchain_tavily.TavilySearch") as mock_tavily_class,
    ):
        mock_settings.return_value.search_provider = "tavily"
        mock_settings.return_value.tavily_api_key = "tvly-test-key"
        mock_tool = mock_tavily_class.return_value
        mock_tool.invoke.return_value = mock_tavily_result

        result = search_tool.invoke({"query": "test query"})

        assert "Test Result" in result
        assert "Test content here" in result
        assert "Source: https://example.com" in result
        mock_tavily_class.assert_called_once_with(max_results=5, topic="general")
        mock_tool.invoke.assert_called_once_with({"query": "test query"})


@pytest.mark.asyncio
async def test_document_lookup_tool_found():
    """Document lookup returns content when document exists."""

    async def get_doc(doc_id: str, tenant_id: str):
        if doc_id == "doc1" and tenant_id == "t1":
            return {"title": "Test Doc", "text": "Content here"}
        return None

    tool = create_document_lookup_tool("t1", get_doc)
    result = await tool.ainvoke({"document_id": "doc1"})
    assert "Test Doc" in result
    assert "Content here" in result


@pytest.mark.asyncio
async def test_document_lookup_tool_not_found():
    """Document lookup returns not found when document missing."""

    async def get_doc(doc_id: str, tenant_id: str):
        return None

    tool = create_document_lookup_tool("t1", get_doc)
    result = await tool.ainvoke({"document_id": "missing"})
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_document_lookup_tool_error_on_fetch():
    """Document lookup returns error message when get_document_fn raises."""

    async def get_doc(doc_id: str, tenant_id: str):
        raise ValueError("Database connection failed")

    tool = create_document_lookup_tool("t1", get_doc)
    result = await tool.ainvoke({"document_id": "doc1"})
    assert "Error" in result
    assert "Database connection failed" in result
