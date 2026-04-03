"""Base error types for the application."""

from __future__ import annotations

from typing import Any, Optional


class AppError(Exception):
    """Base exception for all application errors.

    This is the root exception type for the platform. All domain,
    service, and infrastructure errors should inherit from this.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error code for programmatic handling
        status_code: HTTP status code (400, 403, 404, 500, 503, etc.)
        details: Additional error context as dictionary
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize an AppError.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code (e.g., "VALIDATION_ERROR")
            status_code: HTTP status code (400, 404, 500, etc.)
            details: Additional error context as dict
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dict for API responses.

        Returns:
            Dictionary representation of the error suitable for JSON serialization
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(AppError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class NotFoundError(AppError):
    """Raised when a requested resource is not found."""

    def __init__(
        self, message: str, resource_type: str = "resource", resource_id: str = ""
    ) -> None:
        super().__init__(
            message,
            error_code="NOT_FOUND",
            status_code=404,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )


class AuthorizationError(AppError):
    """Raised when user lacks permission for an operation."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
        )


class ConflictError(AppError):
    """Raised when operation conflicts with existing state."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            error_code="CONFLICT",
            status_code=409,
            details=details,
        )


class ServiceUnavailableError(AppError):
    """Raised when a required service is unavailable."""

    def __init__(
        self, message: str, service_name: str = "", details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details or {"service": service_name},
        )
