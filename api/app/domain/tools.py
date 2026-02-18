"""Domain interfaces for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ToolExecutionResult:
    """Result of a tool execution."""

    status: ToolExecutionStatus
    output: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class ITool(ABC):
    """Interface for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        pass
