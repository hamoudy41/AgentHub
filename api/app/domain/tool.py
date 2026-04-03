"""Tool domain model: definition, input, output, execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.core.types import ToolID


class ToolType(str, Enum):
    """Tool classification for different execution contexts."""

    FUNCTION = "function"
    API = "api"
    CALCULATOR = "calculator"
    SEARCH = "search"
    DOCUMENT_LOOKUP = "document_lookup"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ToolInput:
    """Structured input to a tool."""

    name: str
    value: Any
    description: Optional[str] = None


@dataclass(frozen=True)
class ToolOutput:
    """Structured output from a tool."""

    success: bool
    value: Optional[Any] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool: metadata, parameters, behavior."""

    id: ToolID
    name: str
    description: str
    tool_type: ToolType
    input_schema: dict[str, Any]  # JSON schema for inputs
    output_schema: Optional[dict[str, Any]] = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    """A callable tool with definition and execution context."""

    definition: ToolDefinition
    callable: Optional[Any] = None  # Actual function/method

    async def execute(self, inputs: dict[str, Any]) -> ToolOutput:
        """Execute the tool with given inputs.

        This is typically overridden in concrete implementations.

        Args:
            inputs: Dictionary of input parameters for the tool

        Returns:
            ToolOutput with success status and result or error
        """
        if not self.callable:
            return ToolOutput(success=False, error="Tool not callable")

        try:
            # Check if callable is coroutine function
            import inspect

            if inspect.iscoroutinefunction(self.callable):
                result = await self.callable(**inputs)
            else:
                result = self.callable(**inputs)
            return ToolOutput(success=True, value=result)
        except Exception as exc:
            error_msg = str(exc)
            return ToolOutput(success=False, error=error_msg)
