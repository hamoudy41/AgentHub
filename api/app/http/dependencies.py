from __future__ import annotations

from fastapi import Header

from app.core.config import Settings


def build_tenant_dependency(settings: Settings):
    async def get_tenant_id(
        x_tenant_id: str | None = Header(default=None, alias=settings.tenant_header_name),
    ) -> str:
        return x_tenant_id or settings.default_tenant_id

    return get_tenant_id
