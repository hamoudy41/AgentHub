"""Audit repository: CRUD for audit logs."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AiCallAudit
from app.core.types import TenantID
from app.core.logging import get_logger

from .base import Repository

logger = get_logger(__name__)


class AuditRepository(Repository[AiCallAudit]):
    """Repository for audit log persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, entity: AiCallAudit) -> AiCallAudit:
        """Create a new audit record."""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def read(self, id: str, tenant_id: Optional[TenantID] = None) -> Optional[AiCallAudit]:
        """Read an audit record by ID, scoped to tenant."""
        if tenant_id is None:
            return None
        result = await self.session.execute(
            select(AiCallAudit).where(
                AiCallAudit.id == id,
                AiCallAudit.tenant_id == str(tenant_id),
            )
        )
        return result.scalar_one_or_none()

    async def update(self, entity: AiCallAudit) -> AiCallAudit:
        """Update an audit record."""
        await self.session.merge(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, id: str, tenant_id: Optional[TenantID] = None) -> bool:
        """Delete an audit record by ID, scoped to tenant."""
        if tenant_id is None:
            return False
        result = await self.session.execute(
            delete(AiCallAudit).where(
                AiCallAudit.id == id,
                AiCallAudit.tenant_id == str(tenant_id),
            )
        )
        await self.session.commit()
        return result.rowcount > 0

    async def list(self, tenant_id: TenantID, **filters: dict) -> list[AiCallAudit]:
        """List audit records for a tenant."""
        stmt = select(AiCallAudit).where(AiCallAudit.tenant_id == str(tenant_id))

        if "flow_name" in filters:
            stmt = stmt.where(AiCallAudit.flow_name == filters["flow_name"])

        if "success_only" in filters and filters["success_only"]:
            stmt = stmt.where(AiCallAudit.success)

        if "created_after" in filters:
            stmt = stmt.where(AiCallAudit.created_at >= filters["created_after"])

        # Order by created_at descending
        stmt = stmt.order_by(AiCallAudit.created_at.desc())

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def purge_older_than(self, tenant_id: Optional[TenantID], cutoff: datetime) -> int:
        """Delete audit records older than cutoff date.

        If tenant_id is provided, only purge for that tenant.
        Otherwise, purge across all tenants.
        """
        stmt = delete(AiCallAudit).where(AiCallAudit.created_at < cutoff)

        if tenant_id:
            stmt = stmt.where(AiCallAudit.tenant_id == str(tenant_id))

        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount
