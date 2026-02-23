"""Calculator tool for the agent."""

from __future__ import annotations

import ast
import math
import operator

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

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 1000
_MAX_INT_BITS = 4096  # Prevent extremely large integer allocations


def _safe_eval(expr: str) -> float | int:
    """Evaluate a simple math expression using a restricted AST."""
    tree = ast.parse(expr, mode="eval")

    def _eval_node(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numbers are allowed")
            return node.value

        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            value = _eval_node(node.operand)
            result = op(value)
            if isinstance(result, bool) or not isinstance(result, (int, float)):
                raise ValueError("Result is not a real number")
            return result

        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")

            if isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_POWER_EXPONENT:
                    raise ValueError("Exponent too large")
                if (
                    isinstance(left, int)
                    and isinstance(right, int)
                    and right >= 0
                    and abs(left) > 1
                ):
                    estimated_bits = right * abs(left).bit_length()
                    if estimated_bits > _MAX_INT_BITS:
                        raise ValueError("Result too large")

            result = op(left, right)
            if isinstance(result, bool) or not isinstance(result, (int, float)):
                raise ValueError("Result is not a real number")
            if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
                raise ValueError("Result too large")
            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("Result is not finite")
            return result

        raise ValueError(f"Unsupported expression: {type(node)}")

    return _eval_node(tree)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression (numbers, parentheses, + - * / ** // %)."""
    try:
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression."
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            return f"Error: Expression too long (>{_MAX_EXPRESSION_LENGTH} chars)."

        result = _safe_eval(expression)
        return str(result)
    except ZeroDivisionError as e:
        return f"Error: Division by zero - {e}"
    except SyntaxError as e:
        return f"Error: Invalid syntax in expression - {e}"
    except ValueError as e:
        return f"Error: Unsupported operation - {e}"
    except Exception as e:
        return f"Error: {e}"
