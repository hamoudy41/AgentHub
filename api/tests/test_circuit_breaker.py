"""Tests for circuit breaker functionality."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)


def test_circuit_breaker_starts_closed():
    """Test that circuit breaker starts in CLOSED state."""
    cb = CircuitBreaker("test")
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_circuit_breaker_can_execute_when_closed():
    """Test that requests can execute when circuit is CLOSED."""
    cb = CircuitBreaker("test")
    can_execute, reason = cb.can_execute()
    assert can_execute is True
    assert reason is None


def test_circuit_breaker_opens_after_threshold_failures():
    """Test that circuit opens after reaching failure threshold."""
    config = CircuitBreakerConfig(failure_threshold=3)
    cb = CircuitBreaker("test", config)
    
    # Record failures
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 1
    
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.failure_count == 3


def test_circuit_breaker_blocks_requests_when_open():
    """Test that circuit breaker blocks requests when OPEN."""
    config = CircuitBreakerConfig(failure_threshold=1)
    cb = CircuitBreaker("test", config)
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    can_execute, reason = cb.can_execute()
    assert can_execute is False
    assert reason == "Circuit test is OPEN"


def test_circuit_breaker_transitions_to_half_open_after_timeout():
    """Test that circuit transitions to HALF_OPEN after recovery timeout."""
    config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
    cb = CircuitBreaker("test", config)
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(0.15)
    
    can_execute, reason = cb.can_execute()
    assert can_execute is True
    assert cb.state == CircuitState.HALF_OPEN


def test_circuit_breaker_closes_after_successful_calls_in_half_open():
    """Test that circuit closes after success threshold in HALF_OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1,
        success_threshold=2,
    )
    cb = CircuitBreaker("test", config)
    
    # Open the circuit
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    # Transition to HALF_OPEN
    time.sleep(0.15)
    cb.can_execute()
    assert cb.state == CircuitState.HALF_OPEN
    
    # Record successes
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.success_count == 1
    
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.success_count == 0


def test_circuit_breaker_reopens_on_failure_in_half_open():
    """Test that circuit reopens on failure in HALF_OPEN state."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1,
    )
    cb = CircuitBreaker("test", config)
    
    # Open the circuit
    cb.record_failure()
    
    # Transition to HALF_OPEN
    time.sleep(0.15)
    cb.can_execute()
    assert cb.state == CircuitState.HALF_OPEN
    
    # Record failure in HALF_OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_resets_failure_count_on_success_in_closed():
    """Test that failure count resets on success in CLOSED state."""
    config = CircuitBreakerConfig(failure_threshold=5)
    cb = CircuitBreaker("test", config)
    
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2
    
    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_get_state():
    """Test that get_state returns correct information."""
    cb = CircuitBreaker("test_circuit")
    
    state = cb.get_state()
    assert state["name"] == "test_circuit"
    assert state["state"] == CircuitState.CLOSED
    assert state["failure_count"] == 0
    assert state["success_count"] == 0
    assert state["last_failure_time"] is None


def test_circuit_breaker_get_state_after_failure():
    """Test get_state after recording a failure."""
    cb = CircuitBreaker("test")
    
    cb.record_failure()
    state = cb.get_state()
    
    assert state["failure_count"] == 1
    assert state["last_failure_time"] is not None


def test_circuit_breaker_custom_config():
    """Test circuit breaker with custom configuration."""
    config = CircuitBreakerConfig(
        failure_threshold=10,
        recovery_timeout=60.0,
        success_threshold=5,
        timeout_seconds=30.0,
    )
    cb = CircuitBreaker("custom", config)
    
    assert cb.config.failure_threshold == 10
    assert cb.config.recovery_timeout == 60.0
    assert cb.config.success_threshold == 5
    assert cb.config.timeout_seconds == 30.0


@patch("app.circuit_breaker.logger")
def test_circuit_breaker_logs_state_transitions(mock_logger):
    """Test that circuit breaker logs state transitions."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1,
        success_threshold=1,
    )
    cb = CircuitBreaker("test", config)
    
    # Open the circuit
    cb.record_failure()
    mock_logger.error.assert_called_once()
    
    # Transition to HALF_OPEN
    time.sleep(0.15)
    cb.can_execute()
    mock_logger.info.assert_called()
    
    # Close the circuit
    cb.record_success()
    assert mock_logger.info.call_count == 2


def test_circuit_state_to_metric_conversion():
    """Test _circuit_state_to_metric conversion function."""
    from app.circuit_breaker import _circuit_state_to_metric
    
    assert _circuit_state_to_metric(CircuitState.CLOSED) == 0
    assert _circuit_state_to_metric(CircuitState.HALF_OPEN) == 1
    assert _circuit_state_to_metric(CircuitState.OPEN) == 2
