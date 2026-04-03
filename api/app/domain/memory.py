"""Memory domain model: state and history storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from app.core.types import AgentID, TenantID


class MemoryType(str, Enum):
    """Type of memory stored (short-term, long-term, etc.)."""

    SHORT_TERM = "short_term"  # Conversation history
    LONG_TERM = "long_term"  # Persistent facts
    CONTEXT = "context"  # Execution context
    PREFERENCE = "preference"  # User preferences


@dataclass
class MemoryEntry:
    """A single memory entry (fact, history item, etc.)."""

    key: str
    value: Any
    memory_type: MemoryType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


@dataclass
class Memory:
    """Agent memory: storage for state, history, and facts."""

    tenant_id: TenantID
    agent_id: AgentID
    entries: dict[str, MemoryEntry] = field(default_factory=dict)

    def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        expires_at: Optional[datetime] = None,
    ) -> None:
        """Store a value in memory.

        Args:
            key: Unique key for this memory entry
            value: Value to store (any serializable object)
            memory_type: Type of memory (short-term, long-term, etc.)
            expires_at: Optional expiration datetime
        """
        self.entries[key] = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            expires_at=expires_at,
        )

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory.

        Returns None if not found or expired.

        Args:
            key: Memory key to retrieve

        Returns:
            The stored value or None if not found/expired
        """
        entry = self.entries.get(key)
        if not entry:
            return None
        if entry.is_expired:
            del self.entries[key]
            return None
        return entry.value

    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """Clear memory entries by type or all."""
        if memory_type:
            self.entries = {k: v for k, v in self.entries.items() if v.memory_type != memory_type}
        else:
            self.entries.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries; return count removed."""
        before = len(self.entries)
        self.entries = {k: v for k, v in self.entries.items() if not v.is_expired}
        return before - len(self.entries)
