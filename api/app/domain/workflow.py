"""Workflow domain model: multi-step task execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from app.core.types import TenantID

from .task import TaskResult


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A step in a workflow (maps to a task)."""

    name: str
    action: str
    inputs: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Execution of a workflow."""

    workflow_id: str
    tenant_id: TenantID
    status: WorkflowStatus = WorkflowStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        """Execution duration in milliseconds."""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def add_task_result(self, task_result: TaskResult) -> None:
        """Record the result of a task."""
        self.task_results[str(task_result.task_id)] = task_result


@dataclass
class Workflow:
    """Definition of a workflow: sequence of steps."""

    id: str
    name: str
    description: str
    steps: list[WorkflowStep]
    tenant_id: TenantID
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
