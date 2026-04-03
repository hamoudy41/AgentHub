"""Agent domain model: autonomous entity for task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from app.core.types import AgentID, TenantID

from .memory import Memory
from .tool import Tool


class AgentState(str, Enum):
    """State of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    name: str
    description: str
    model: str  # LLM model to use
    temperature: float = 0.7
    max_iterations: int = 10
    timeout_seconds: float = 60.0
    enable_memory: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTool:
    """A tool available to an agent."""

    tool: Tool
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecution:
    """Result of a single agent execution."""

    agent_id: AgentID
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, succeeded, failed
    output: Optional[Any] = None
    error: Optional[str] = None
    iterations: int = 0
    tool_calls_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        """Execution duration in milliseconds."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


@dataclass
class Agent:
    """Autonomous agent for task execution.

    An agent is an autonomous entity that can:
    - Execute tasks using available tools
    - Maintain state via memory
    - Make decisions based on context

    Attributes:
        id: Unique agent identifier
        tenant_id: Tenant this agent belongs to
        config: Agent configuration (model, temperature, etc.)
        state: Current state (idle, running, paused, stopped)
        tools: Tools available to this agent
        memory: Optional memory store for agent state
        created_at: Timestamp when agent was created
        metadata: Additional agent metadata
    """

    id: AgentID
    tenant_id: TenantID
    config: AgentConfig
    state: AgentState = AgentState.IDLE
    tools: dict[str, AgentTool] = field(default_factory=dict)
    memory: Optional[Memory] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tool(self, tool: Tool, required: bool = False) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools[tool.definition.name] = AgentTool(tool=tool, required=required)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent's toolkit."""
        self.tools.pop(tool_name, None)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        agent_tool = self.tools.get(tool_name)
        return agent_tool.tool if agent_tool else None
