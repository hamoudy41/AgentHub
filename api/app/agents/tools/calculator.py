"""Calculator tool for the agent."""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(expr: str) -> float | int:
    """Evaluate a simple math expression (numbers and +, -, *, /, ** only)."""
    tree = ast.parse(expr, mode="eval")
    if not isinstance(tree.body, ast.BinOp):

        def _eval_node(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -_eval_node(node.operand)
            if isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                op = _BIN_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op)}")
                return op(left, right)
            raise ValueError(f"Unsupported node: {type(node)}")

        return _eval_node(tree.body)

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        raise ValueError(f"Unsupported node: {type(node)}")

    return _eval_node(tree.body)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic like 2+3*4, 100/5, 2**10. For average of numbers, use (a+b+c)/n, e.g. (1+2+5+6)/4."""
    try:
        result = _safe_eval(expression.strip())
        return str(result)
    except Exception as e:
        return f"Error: {e}"
