"""
AI call audit retention: purge records older than AI_AUDIT_RETENTION_DAYS.
Run via cron: python -m app.audit
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from .core.config import get_settings
from .core.logging import get_logger
from .db import get_engine
from .models import AiCallAudit

logger = get_logger(__name__)


async def purge_expired_audits() -> int:
    """Delete ai_call_audit records older than ai_audit_retention_days. Returns count deleted."""
    settings = get_settings()
    days = settings.ai_audit_retention_days
    if days <= 0:
        logger.info("audit.purge_skipped", reason="retention_days_disabled")
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(delete(AiCallAudit).where(AiCallAudit.created_at < cutoff))
        deleted = result.rowcount
    logger.info("audit.purge_complete", deleted=deleted, cutoff=cutoff.isoformat())
    return deleted


def main() -> None:
    asyncio.run(purge_expired_audits())


if __name__ == "__main__":
    main()
