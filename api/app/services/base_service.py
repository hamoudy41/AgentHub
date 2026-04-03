"""Base service class with common functionality."""

from __future__ import annotations

from abc import ABC

from app.core.logging import get_logger
from app.core.context import get_execution_context


class BaseService(ABC):
    """Base class for all services.

    Provides common functionality like logging and context handling.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = get_logger(f"service.{name}")

    def get_context(self):
        """Get current execution context."""
        return get_execution_context()

    def log_info(self, message: str, **kwargs) -> None:
        """Log info level message with context."""
        self.logger.info(message, service=self.name, **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error level message with context."""
        self.logger.error(message, service=self.name, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning level message with context."""
        self.logger.warning(message, service=self.name, **kwargs)
