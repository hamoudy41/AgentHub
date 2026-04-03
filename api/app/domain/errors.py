"""Domain-specific error types."""

from __future__ import annotations

from app.core.errors import AppError


class DomainError(AppError):
    """Base error for domain logic violations."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code="DOMAIN_ERROR",
            status_code=400,
            details=details,
        )


class InvalidToolError(DomainError):
    """Raised when a tool is invalid or not found."""

    pass


class AgentExecutionError(DomainError):
    """Raised when agent execution fails."""

    pass


class WorkflowExecutionError(DomainError):
    """Raised when workflow execution fails."""

    pass
