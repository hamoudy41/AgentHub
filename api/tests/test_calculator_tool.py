"""Tests for calculator_tool safety and basic behavior."""

from __future__ import annotations

from app.agents.tools.calculator import calculator_tool


def test_calculator_tool_basic_arithmetic():
    assert calculator_tool.invoke({"expression": "2+2"}) == "4"


def test_calculator_tool_empty_expression():
    assert calculator_tool.invoke({"expression": "   "}).startswith("Error:")


def test_calculator_tool_rejects_non_numeric_constants():
    out = calculator_tool.invoke({"expression": "True + 1"})
    assert out.startswith("Error:")


def test_calculator_tool_rejects_huge_integer_pow():
    out = calculator_tool.invoke({"expression": "999999999999**1000"})
    assert "too large" in out.lower()


def test_calculator_tool_rejects_non_finite_results():
    out = calculator_tool.invoke({"expression": "1e308*1e308"})
    assert "not finite" in out.lower()
