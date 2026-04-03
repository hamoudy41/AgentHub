"""Tool management and execution service."""

from __future__ import annotations

from typing import Any, Callable, Optional

from app.core.context import ExecutionContext, get_execution_context
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.domain.tool import Tool, ToolDefinition, ToolType

from .base_service import BaseService

logger = get_logger(__name__)


class ToolService(BaseService):
    """Service for managing tool registry and execution.

    Provides tool discovery, validation, and execution with proper error handling.
    Supports dynamic tool registration and plugin patterns.

    Args:
        description: Service description for logging
    """

    def __init__(self) -> None:
        """Initialize tool service."""
        super().__init__("tool")
        # Registry: tool_id -> Tool
        self._registry: dict[str, Tool] = {}

    def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        execute_fn: Callable[[dict[str, Any]], Any],
        *,
        tool_type: ToolType = ToolType.FUNCTION,
    ) -> None:
        """Register a tool in the registry.

        Args:
            tool_id: Unique tool identifier
            name: Human-readable tool name
            description: Tool description for LLM context
            input_schema: JSON schema for input validation
            execute_fn: Async function to execute
            tool_type: Type of tool (FUNCTION, API, CALCULATOR, etc.)

        Raises:
            ValidationError: If tool_id already registered
        """
        if tool_id in self._registry:
            self.log_warning("tool.already_registered", tool_id=tool_id)
            raise ValidationError(f"Tool {tool_id} already registered")

        definition = ToolDefinition(
            id=tool_id,
            name=name,
            description=description,
            input_schema=input_schema,
            tool_type=tool_type,
        )
        tool = Tool(definition=definition, callable=execute_fn)
        self._registry[tool_id] = tool
        self.log_info(
            "tool.registered",
            tool_id=tool_id,
            name=name,
            tool_type=tool_type.value,
        )

    def get_tool(self, tool_id: str) -> Tool:
        """Retrieve tool by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool instance

        Raises:
            NotFoundError: If tool not found
        """
        if tool_id not in self._registry:
            self.log_warning("tool.not_found", tool_id=tool_id)
            raise NotFoundError(f"Tool {tool_id} not found")
        return self._registry[tool_id]

    def list_tools(
        self,
        *,
        tool_type: Optional[ToolType] = None,
    ) -> list[ToolDefinition]:
        """List available tools with optional filtering.

        Args:
            tool_type: Optional filter by tool type

        Returns:
            List of tool definitions
        """
        tools = list(self._registry.values())
        if tool_type:
            tools = [t for t in tools if t.definition.tool_type == tool_type]

        self.log_info(
            "tool.list",
            count=len(tools),
            filter_type=tool_type.value if tool_type else None,
        )
        return [t.definition for t in tools]

    async def execute_tool(
        self,
        tool_id: str,
        inputs: dict[str, Any],
        *,
        context: ExecutionContext | None = None,
    ) -> Any:
        """Execute a tool with given inputs.

        Args:
            tool_id: Tool identifier
            inputs: Input arguments for tool
            context: Optional execution context (uses current if not provided)

        Returns:
            Tool output/result

        Raises:
            NotFoundError: If tool not found
            Exception: If tool execution raises
        """
        ctx = context or get_execution_context()
        tool = self.get_tool(tool_id)

        self.log_info(
            "tool.execute_started",
            tool_id=tool_id,
            tool_name=tool.definition.name,
            tenant_id=ctx.tenant_id,
        )

        try:
            result = await tool.execute(inputs)
            self.log_info(
                "tool.execute_success",
                tool_id=tool_id,
                tenant_id=ctx.tenant_id,
            )
            return result
        except Exception as exc:
            self.log_error(
                "tool.execute_failed",
                tool_id=tool_id,
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    def get_tools_for_agent(
        self,
        tool_ids: list[str],
    ) -> list[Tool]:
        """Get multiple tools by ID list.

        Args:
            tool_ids: List of tool identifiers

        Returns:
            List of Tool instances (skips missing tools with warning)
        """
        tools = []
        for tool_id in tool_ids:
            try:
                tools.append(self.get_tool(tool_id))
            except NotFoundError:
                self.log_warning("tool.missing_in_list", tool_id=tool_id)

        self.log_info(
            "tool.prepare_agent_tools",
            requested=len(tool_ids),
            found=len(tools),
        )
        return tools

    def unregister_tool(self, tool_id: str) -> None:
        """Unregister a tool from the registry.

        Args:
            tool_id: Tool identifier

        Raises:
            NotFoundError: If tool not found
        """
        if tool_id not in self._registry:
            raise NotFoundError(f"Tool {tool_id} not found")
        del self._registry[tool_id]
        self.log_info("tool.unregistered", tool_id=tool_id)
