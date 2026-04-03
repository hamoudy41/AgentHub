"""Audit logging service."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


from app.core.context import ExecutionContext, get_execution_context
from app.core.logging import get_logger
from app.models import AiCallAudit
from app.persistence.repositories.audit import AuditRepository

from .base_service import BaseService

logger = get_logger(__name__)


class AuditService(BaseService):
    """Service for recording and managing audit logs.

    Tracks all AI flow executions, tool calls, and significant events
    for compliance, debugging, and optimization.

    Args:
        repository: AuditRepository for data access
    """

    def __init__(self, repository: AuditRepository) -> None:
        """Initialize audit service.

        Args:
            repository: AuditRepository instance
        """
        super().__init__("audit")
        self._repository = repository

    async def record_flow_execution(
        self,
        flow_name: str,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        success: bool,
        context: ExecutionContext | None = None,
    ) -> str:
        """Record execution of an AI flow.

        Args:
            flow_name: Name of the flow (ask, classify, notary, etc.)
            request: Request payload
            response: Response payload
            success: Whether execution succeeded
            context: Optional execution context (uses current if not provided)

        Returns:
            Audit record ID
        """
        ctx = context or get_execution_context()
        self.log_info(
            "audit.record_flow",
            flow=flow_name,
            success=success,
            request_keys=list(request.keys()),
            tenant_id=ctx.tenant_id,
        )

        # Create audit log record
        record_id = str(uuid.uuid4())
        log = AiCallAudit(
            id=record_id,
            tenant_id=ctx.tenant_id,
            flow_name=flow_name,
            request_payload=request,
            response_payload=response,
            success=success,
            created_at=datetime.now(timezone.utc),
        )
        result = await self._repository.create(log)
        return result.id

    async def record_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        input_args: dict[str, Any],
        output: Any,
        *,
        success: bool,
        latency_ms: float,
        context: ExecutionContext | None = None,
    ) -> None:
        """Record execution of a tool call.

        Args:
            agent_id: ID of agent calling tool
            tool_name: Name of tool
            input_args: Input arguments passed to tool
            output: Output/result from tool
            success: Whether tool call succeeded
            latency_ms: Time taken to execute tool
            context: Optional execution context (uses current if not provided)
        """
        await asyncio.sleep(0)
        ctx = context or get_execution_context()
        self.log_info(
            "audit.tool_call",
            agent_id=agent_id,
            tool=tool_name,
            success=success,
            latency_ms=latency_ms,
            input_keys=list(input_args.keys()),
            output_type=type(output).__name__,
            tenant_id=ctx.tenant_id,
        )

    async def purge_old_records(
        self,
        retention_days: int,
        *,
        context: ExecutionContext | None = None,
    ) -> int:
        """Delete audit records older than retention period.

        Args:
            retention_days: Keep records from last N days (delete older)
            context: Optional execution context (uses current if not provided)

        Returns:
            Number of records deleted
        """
        ctx = context or get_execution_context()
        if retention_days <= 0:
            self.log_info("audit.purge_disabled", context=ctx)
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        deleted = await self._repository.purge_older_than(ctx.tenant_id, cutoff)
        self.log_info(
            "audit.purged",
            count=deleted,
            retention_days=retention_days,
            tenant_id=ctx.tenant_id,
        )
        return deleted

    async def get_flow_stats(
        self,
        flow_name: str,
        *,
        days: int = 7,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Get statistics for a flow over recent period.

        Args:
            flow_name: Flow to analyze
            days: Look back this many days
            context: Optional execution context (uses current if not provided)

        Returns:
            Stats dict with success_count, failure_count, success_rate
        """
        ctx = context or get_execution_context()
        since = datetime.now(timezone.utc) - timedelta(days=days)

        # Get records from repository
        all_records = await self._repository.list(
            tenant_id=ctx.tenant_id,
            flow_name=flow_name,
            created_after=since,
        )

        success_count = sum(1 for r in all_records if r.success)
        failure_count = len(all_records) - success_count
        total = len(all_records)
        success_rate = (success_count / total * 100) if total > 0 else 0.0

        stats = {
            "flow": flow_name,
            "period_days": days,
            "total_executions": total,
            "successes": success_count,
            "failures": failure_count,
            "success_rate_percent": round(success_rate, 2),
        }
        self.log_info(
            "audit.flow_stats",
            flow=flow_name,
            stats=stats,
            tenant_id=ctx.tenant_id,
        )
        return stats
