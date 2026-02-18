from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# HTTP request metrics
REQUEST_COUNT = Counter(
    "ai_platform_requests_total", "Total requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "ai_platform_request_duration_seconds",
    "Request latency",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# LLM call metrics
LLM_CALLS = Counter("ai_platform_llm_calls_total", "LLM API calls", ["flow", "source"])

LLM_LATENCY = Histogram(
    "ai_platform_llm_duration_seconds",
    "LLM request latency in seconds",
    ["provider", "flow"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

LLM_ERRORS = Counter(
    "ai_platform_llm_errors_total",
    "Total LLM errors",
    ["provider", "error_type"],
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "ai_platform_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["circuit_name"],
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "ai_platform_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["circuit_name"],
)

# Agent metrics
AGENT_EXECUTIONS = Counter(
    "ai_platform_agent_executions_total",
    "Total agent executions",
    ["tenant_id", "status"],  # status: success/error/timeout
)

AGENT_DURATION = Histogram(
    "ai_platform_agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    ["tenant_id"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

AGENT_TOOL_CALLS = Counter(
    "ai_platform_agent_tool_calls_total",
    "Total tool calls by agent",
    ["tool_name", "status"],  # status: success/error
)

AGENT_FALLBACKS = Counter(
    "ai_platform_agent_fallbacks_total",
    "Total agent fallbacks to search",
    ["reason"],  # reason: malformed/empty/error
)

# Security metrics
SECURITY_VALIDATIONS = Counter(
    "ai_platform_security_validations_total",
    "Total security validations",
    ["validation_type", "result"],  # type: injection_check/input_size, result: pass/fail
)

SECURITY_BLOCKS = Counter(
    "ai_platform_security_blocks_total",
    "Total requests blocked by security",
    ["reason"],  # reason: injection/too_long/malicious
)


def get_metrics() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
