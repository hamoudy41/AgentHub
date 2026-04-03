"""Agent orchestration and execution service."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.context import ExecutionContext, get_execution_context
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.domain.agent import Agent, AgentConfig, AgentState
from app.domain.memory import Memory

from .base_service import BaseService
from .memory_service import MemoryService
from .tool_service import ToolService

logger = get_logger(__name__)


class AgentService(BaseService):
    """Service for agent lifecycle and execution.

    Manages agent creation, configuration, tool management, and execution.
    Provides a high-level interface to agent runtime orchestration.

    Args:
        tool_service: ToolService for managing agent tools
        memory_service: MemoryService for agent memory
    """

    def __init__(
        self,
        tool_service: ToolService,
        memory_service: MemoryService,
    ) -> None:
        """Initialize agent service.

        Args:
            tool_service: ToolService instance
            memory_service: MemoryService instance
        """
        super().__init__("agent")
        self._tool_service = tool_service
        self._memory_service = memory_service
        # Agent registry: agent_id -> Agent
        self._agents: dict[str, Agent] = {}

    def create_agent(
        self,
        agent_id: str,
        name: str,
        model: str,
        *,
        temperature: float = 0.7,
        max_iterations: int = 10,
        timeout: float = 30.0,
        context: ExecutionContext | None = None,
    ) -> Agent:
        """Create a new agent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            model: LLM model to use
            temperature: Temperature for LLM sampling (0-1)
            max_iterations: Max steps in agent loop
            timeout: Execution timeout in seconds
            context: Optional execution context (uses current if not provided)

        Returns:
            Created Agent instance
        """
        ctx = context or get_execution_context()
        self.log_info(
            "agent.create_started",
            agent_id=agent_id,
            name=name,
            model=model,
            tenant_id=ctx.tenant_id,
        )

        config = AgentConfig(
            name=name,
            description=f"Agent {name}",
            model=model,
            temperature=temperature,
            max_iterations=max_iterations,
            timeout_seconds=timeout,
        )
        agent = Agent(
            id=agent_id,
            tenant_id=ctx.tenant_id,
            config=config,
            state=AgentState.IDLE,
            memory=Memory(tenant_id=ctx.tenant_id, agent_id=agent_id),
        )
        self._agents[agent_id] = agent

        self.log_info(
            "agent.created",
            agent_id=agent_id,
            tenant_id=ctx.tenant_id,
        )
        return agent

    def get_agent(
        self,
        agent_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> Agent:
        """Retrieve an agent by ID.

        Args:
            agent_id: Agent identifier
            context: Optional execution context (uses current if not provided)

        Returns:
            Agent instance

        Raises:
            ValueError: If agent not found
        """
        ctx = context or get_execution_context()
        if agent_id not in self._agents:
            self.log_warning(
                "agent.not_found",
                agent_id=agent_id,
                tenant_id=ctx.tenant_id,
            )
            raise NotFoundError(f"Agent {agent_id} not found")
        return self._agents[agent_id]

    async def add_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Add a tool to an agent.

        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            context: Optional execution context (uses current if not provided)

        Raises:
            NotFoundError: If agent or tool not found
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        agent = self.get_agent(agent_id, context=ctx)
        tool = self._tool_service.get_tool(tool_id)

        agent.add_tool(tool)
        self.log_info(
            "agent.tool_added",
            agent_id=agent_id,
            tool_id=tool_id,
            tenant_id=ctx.tenant_id,
        )

    async def remove_tool_from_agent(
        self,
        agent_id: str,
        tool_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Remove a tool from an agent.

        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            context: Optional execution context (uses current if not provided)

        Raises:
            NotFoundError: If agent not found
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        agent = self.get_agent(agent_id, context=ctx)
        tool = self._tool_service.get_tool(tool_id)
        agent.remove_tool(tool.definition.name)
        self.log_info(
            "agent.tool_removed",
            agent_id=agent_id,
            tool_id=tool_id,
            tenant_id=ctx.tenant_id,
        )

    def get_agent_tools(
        self,
        agent_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> list[Any]:
        """Get all tools assigned to an agent.

        Args:
            agent_id: Agent identifier
            context: Optional execution context (uses current if not provided)

        Returns:
            List of Tool instances
        """
        ctx = context or get_execution_context()
        agent = self.get_agent(agent_id, context=ctx)
        return [agent_tool.tool for agent_tool in agent.tools.values()]

    async def execute_agent(
        self,
        agent_id: str,
        task: str,
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute an agent on a task.

        Args:
            agent_id: Agent identifier
            task: Task description or user message
            context: Optional execution context (uses current if not provided)

        Returns:
            Execution result with answer, steps, metadata
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        agent = self.get_agent(agent_id, context=ctx)

        self.log_info(
            "agent.execute_started",
            agent_id=agent_id,
            task=task[:100],
            tenant_id=ctx.tenant_id,
        )

        try:
            # Update agent state
            agent.state = AgentState.RUNNING

            # In full implementation, would call actual agent runtime here.
            # For now, return mock result structure.
            result = {
                "agent_id": agent_id,
                "task": task,
                "status": "completed",
                "result": f"Agent {agent.config.name} processed: {task[:100]}",
                "steps": 1,
                "tools_used": list(agent.tools.keys()),
                "metadata": {
                    "model": agent.config.model,
                    "temperature": agent.config.temperature,
                },
            }

            # Update agent state
            agent.state = AgentState.IDLE

            self.log_info(
                "agent.execute_completed",
                agent_id=agent_id,
                steps=result.get("steps"),
                tenant_id=ctx.tenant_id,
            )
            return result
        except Exception as exc:
            agent.state = AgentState.STOPPED
            self.log_error(
                "agent.execute_failed",
                agent_id=agent_id,
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    def delete_agent(
        self,
        agent_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Delete an agent.

        Args:
            agent_id: Agent identifier
            context: Optional execution context (uses current if not provided)
        """
        ctx = context or get_execution_context()
        if agent_id not in self._agents:
            return

        del self._agents[agent_id]
        self.log_info(
            "agent.deleted",
            agent_id=agent_id,
            tenant_id=ctx.tenant_id,
        )

    async def list_agents(
        self,
        *,
        context: ExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        """List all agents for current tenant.

        Args:
            context: Optional execution context (uses current if not provided)

        Returns:
            List of agent summaries
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        agents = [
            {
                "id": agent.id,
                "name": agent.config.name,
                "model": agent.config.model,
                "state": agent.state.value,
                "tools": list(agent.tools.keys()),
            }
            for agent in self._agents.values()
            if agent.tenant_id == ctx.tenant_id
        ]
        self.log_info(
            "agent.list",
            count=len(agents),
            tenant_id=ctx.tenant_id,
        )
        return agents
