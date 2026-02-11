"""Tests for audit retention and purge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import delete, insert, select

from app.audit import purge_expired_audits
from app.db import get_engine
from app.models import AiCallAudit, Base

pytestmark = pytest.mark.asyncio


async def test_purge_skipped_when_retention_disabled():
    """When ai_audit_retention_days is 0, purge does nothing."""
    with patch("app.audit.get_settings") as mock:
        mock.return_value.ai_audit_retention_days = 0
        deleted = await purge_expired_audits()
    assert deleted == 0


async def test_purge_deletes_old_records():
    """Purge removes ai_call_audit records older than retention days."""
    engine = get_engine()
    old_date = datetime.now(timezone.utc) - timedelta(days=100)
    now = datetime.now(timezone.utc)

    with patch("app.audit.get_settings") as mock:
        mock.return_value.ai_audit_retention_days = 90

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(delete(AiCallAudit))
            await conn.execute(
                insert(AiCallAudit).values(
                    id="old-1",
                    tenant_id="t1",
                    flow_name="ask",
                    request_payload={"q": "x"},
                    response_payload={"a": "y"},
                    success=True,
                    created_at=old_date,
                )
            )
            await conn.execute(
                insert(AiCallAudit).values(
                    id="recent-1",
                    tenant_id="t1",
                    flow_name="ask",
                    request_payload={"q": "y"},
                    response_payload={"a": "z"},
                    success=True,
                    created_at=now,
                )
            )

        deleted = await purge_expired_audits()

    assert deleted == 1
    async with engine.begin() as conn:
        result = await conn.execute(select(AiCallAudit.id))
        remaining = result.scalars().all()
    assert remaining == ["recent-1"]
