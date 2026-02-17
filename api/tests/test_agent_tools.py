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


def test_search_tool_returns_mock_result():
    """Search tool returns mock placeholder."""
    result = search_tool.invoke({"query": "weather Paris"})
    assert "Mock search" in result or "Paris" in result


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
