"""Execution context: tenant, request, audit trail information."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from .types import RequestID, TenantID, UserId

# Thread-local context variable for execution context
_execution_context: contextvars.ContextVar[ExecutionContext | None] = contextvars.ContextVar(
    "execution_context",
    default=None,
)


@dataclass
class ExecutionContext:
    """Rich execution context for tracking requests through the system.

    This object flows through all layers (HTTP -> services -> domain -> persistence)
    and provides consistent metadata for logging, tracing, and auditing.

    Attributes:
        tenant_id: Tenant identifier from request
        request_id: Unique request identifier for tracing
        created_at: Timestamp when context was created
        user_id: Optional user identifier
        metadata: Additional context metadata (custom fields)
    """

    tenant_id: TenantID
    request_id: RequestID
    created_at: datetime
    user_id: Optional[UserId] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls,
        tenant_id: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionContext:
        """Create context from HTTP request headers.

        Args:
            tenant_id: Tenant identifier
            request_id: Request identifier (generated if not provided)
            user_id: User identifier (optional)
            metadata: Additional context metadata
        """
        return cls(
            tenant_id=TenantID(tenant_id),
            request_id=RequestID(request_id or str(uuid4())),
            created_at=datetime.now(timezone.utc),
            user_id=UserId(user_id) if user_id else None,
            metadata=metadata or {},
        )

    def set_context(self) -> None:
        """Set this context as the current execution context."""
        _execution_context.set(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dict for structured logging."""
        return {
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }


def get_execution_context() -> Optional[ExecutionContext]:
    """Get the current execution context from thread-local storage."""
    return _execution_context.get()


def set_execution_context(context: ExecutionContext) -> None:
    """Set the current execution context."""
    context.set_context()


def clear_execution_context() -> None:
    """Clear the current execution context."""
    _execution_context.set(None)
