"""Shared type definitions for the application."""

from __future__ import annotations

from typing import NewType, TypedDict

# Tenant and user identifiers
TenantID = NewType("TenantID", str)
RequestID = NewType("RequestID", str)
UserId = NewType("UserId", str)
ToolID = NewType("ToolID", str)
AgentID = NewType("AgentID", str)
TaskID = NewType("TaskID", str)


class ExecutionMetadata(TypedDict, total=False):
    """Metadata about execution: latency, model, tokens, etc."""

    latency_ms: int
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    provider: str
    cache_hit: bool
    retry_count: int
