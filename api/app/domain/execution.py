"""Execution domain model: results and tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.types import ExecutionMetadata


@dataclass(frozen=True)
class ToolCall:
    """A request to execute a tool."""

    tool_name: str
    inputs: dict[str, Any]
    thought: Optional[str] = None  # Agent's reasoning


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of an execution (agent step, workflow, task, etc.)."""

    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    metadata: ExecutionMetadata = field(default_factory=dict)  # type: ignore
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_tool_call(
        self, tool_name: str, inputs: dict[str, Any], thought: Optional[str] = None
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append(ToolCall(tool_name=tool_name, inputs=inputs, thought=thought))

    def add_tool_result(
        self, tool_name: str, success: bool, output: Any = None, error: str = ""
    ) -> None:
        """Record a tool result."""
        self.tool_results.append(
            ToolResult(
                tool_name=tool_name,
                success=success,
                output=output,
                error=error,
            )
        )
