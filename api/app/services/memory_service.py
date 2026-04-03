"""Agent memory management service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.core.context import ExecutionContext, get_execution_context
from app.core.logging import get_logger
from app.domain.memory import Memory, MemoryType

from .base_service import BaseService

logger = get_logger(__name__)


class MemoryService(BaseService):
    """Service for agent memory management.

    Manages short-term, long-term, context, and preference memory for agents.
    Supports expiration and cleanup of stale entries.

    **Note**: This is an in-memory implementation. For production, integrate
    with Redis or persistent storage.

    Args:
        ttl_seconds: Default time-to-live for memory entries
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Initialize memory service.

        Args:
            ttl_seconds: Default TTL for entries (1 hour)
        """
        super().__init__("memory")
        self._ttl_seconds = ttl_seconds
        # Memory storage: tenant_id -> agent_id -> Memory
        self._storage: dict[str, dict[str, Memory]] = {}

    async def store(
        self,
        agent_id: str,
        memory_type: MemoryType,
        key: str,
        value: Any,
        *,
        ttl_seconds: Optional[int] = None,
        context: ExecutionContext | None = None,
    ) -> None:
        """Store value in agent memory.

        Args:
            agent_id: Agent identifier
            memory_type: Type of memory (SHORT_TERM, LONG_TERM, CONTEXT, PREFERENCE)
            key: Key for storing value
            value: Value to store
            ttl_seconds: Optional time-to-live (uses default if not provided)
            context: Optional execution context (uses current if not provided)
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        # Ensure nested dicts exist
        if ctx.tenant_id not in self._storage:
            self._storage[ctx.tenant_id] = {}
        if agent_id not in self._storage[ctx.tenant_id]:
            self._storage[ctx.tenant_id][agent_id] = Memory(
                tenant_id=ctx.tenant_id,
                agent_id=agent_id,
            )

        memory = self._storage[ctx.tenant_id][agent_id]
        memory.store(key, value, memory_type=memory_type, expires_at=expires_at)
        self.log_info(
            "memory.stored",
            agent=agent_id,
            memory_type=memory_type.value,
            key=key,
            tenant_id=ctx.tenant_id,
        )

    async def retrieve(
        self,
        agent_id: str,
        memory_type: MemoryType,
        key: str,
        *,
        context: ExecutionContext | None = None,
    ) -> Any | None:
        """Retrieve value from agent memory.

        Args:
            agent_id: Agent identifier
            memory_type: Type of memory
            key: Key to retrieve
            context: Optional execution context (uses current if not provided)

        Returns:
            Stored value or None if not found/expired
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()

        if ctx.tenant_id not in self._storage or agent_id not in self._storage[ctx.tenant_id]:
            return None

        memory = self._storage[ctx.tenant_id][agent_id]
        value = memory.retrieve(key)
        if value is not None:
            self.log_info(
                "memory.retrieved",
                agent=agent_id,
                memory_type=memory_type.value,
                key=key,
                tenant_id=ctx.tenant_id,
            )
        return value

    async def clear(
        self,
        agent_id: str,
        memory_type: MemoryType,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Clear all entries in an agent's memory of specific type.

        Args:
            agent_id: Agent identifier
            memory_type: Type of memory to clear
            context: Optional execution context (uses current if not provided)
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()

        if ctx.tenant_id in self._storage and agent_id in self._storage[ctx.tenant_id]:
            memory = self._storage[ctx.tenant_id][agent_id]
            memory.clear(memory_type=memory_type)
            self.log_info(
                "memory.cleared",
                agent=agent_id,
                memory_type=memory_type.value,
                tenant_id=ctx.tenant_id,
            )

    async def cleanup_expired(
        self,
        agent_id: str,
        memory_type: MemoryType,
        *,
        context: ExecutionContext | None = None,
    ) -> int:
        """Remove expired entries from agent memory.

        Args:
            agent_id: Agent identifier
            memory_type: Type of memory to clean
            context: Optional execution context (uses current if not provided)

        Returns:
            Number of entries removed
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()

        if ctx.tenant_id not in self._storage or agent_id not in self._storage[ctx.tenant_id]:
            return 0

        memory = self._storage[ctx.tenant_id][agent_id]
        removed = memory.cleanup_expired()
        if removed > 0:
            self.log_info(
                "memory.expired_entries_removed",
                agent=agent_id,
                memory_type=memory_type.value,
                count=removed,
                tenant_id=ctx.tenant_id,
            )
        return removed

    async def get_agent_summary(
        self,
        agent_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Get summary of all memory for an agent.

        Args:
            agent_id: Agent identifier
            context: Optional execution context (uses current if not provided)

        Returns:
            Dict with memory type -> entry count
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()

        summary = {
            "agent_id": agent_id,
            "tenant_id": ctx.tenant_id,
            "memory_types": {},
        }

        if ctx.tenant_id in self._storage and agent_id in self._storage[ctx.tenant_id]:
            memory = self._storage[ctx.tenant_id][agent_id]
            counts: dict[str, int] = {}
            for entry in memory.entries.values():
                key_name = entry.memory_type.value
                counts[key_name] = counts.get(key_name, 0) + 1
            summary["memory_types"] = counts

        return summary
