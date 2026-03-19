from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import AiCallAudit
from app.security import sanitize_user_input


logger = get_logger(__name__)

LLM_NOT_CONFIGURED_MESSAGE = (
    "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
)


class AiFlowError(Exception):
    """Raised when an AI workflow cannot safely produce a response."""


def sanitize_flow_text(
    value: str,
    *,
    tenant_id: str,
    max_length: int,
    log_event: str,
) -> str:
    try:
        return sanitize_user_input(
            value,
            max_length=max_length,
            check_injection=True,
            tenant_id=tenant_id,
        )
    except ValueError as exc:
        logger.warning(log_event, tenant_id=tenant_id, error=str(exc))
        raise AiFlowError(f"Input validation failed: {exc}") from exc


async def persist_audit_record(
    db: AsyncSession,
    *,
    tenant_id: str,
    flow_name: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    success: bool,
) -> bool:
    audit = AiCallAudit(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        flow_name=flow_name,
        request_payload=request_payload,
        response_payload=response_payload,
        success=success,
    )
    try:
        db.add(audit)
        await db.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.audit_persist_failed", flow=flow_name, error=str(exc))
        try:
            await db.rollback()
        except Exception as rollback_exc:  # noqa: BLE001
            logger.warning("ai_flow.rollback_failed", flow=flow_name, error=str(rollback_exc))
        return False
