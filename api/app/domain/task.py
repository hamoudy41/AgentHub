"""Task domain model: execution unit within workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from app.core.types import TaskID


class TaskStatus(str, Enum):
    """Status of a task in its lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of task execution."""

    task_id: TaskID
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[int]:
        """Duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return self.duration_ms


@dataclass
class Task:
    """A unit of work within a workflow."""

    id: TaskID
    name: str
    action: str  # What to do (e.g., "llm_complete", "tool_execute", "retrieve_documents")
    inputs: dict[str, Any]
    depends_on: list[TaskID] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
