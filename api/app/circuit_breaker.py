"""Circuit breaker pattern for resilient external service calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

from .core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    success_threshold: int = 2  # Successful calls needed to close from half-open
    timeout_seconds: float = 60.0  # Request timeout


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def can_execute(self) -> tuple[bool, str | None]:
        """
        Check if request can be executed.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        if self.state == CircuitState.CLOSED:
            return True, None

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(
                    "circuit_breaker.half_open",
                    name=self.name,
                    failure_count=self.failure_count,
                )
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True, None
            return False, f"Circuit {self.name} is OPEN"

        # HALF_OPEN state
        return True, None

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(
                    "circuit_breaker.closed",
                    name=self.name,
                    success_count=self.success_count,
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_failure_time = None
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success in closed state
            if self.failure_count > 0:
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "circuit_breaker.half_open_failure",
                name=self.name,
            )
            self.state = CircuitState.OPEN
            self.failure_count = self.config.failure_threshold
            return

        self.failure_count += 1
        if self.failure_count >= self.config.failure_threshold:
            logger.error(
                "circuit_breaker.opened",
                name=self.name,
                failure_count=self.failure_count,
                threshold=self.config.failure_threshold,
            )
            self.state = CircuitState.OPEN

    def get_state(self) -> dict[str, any]:
        """Get current circuit breaker state for monitoring."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    pass
