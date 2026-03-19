from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.redis import ping_redis
from app.db import get_db_session
from app.schemas import HealthStatus
from app.services_llm import llm_client


logger = get_logger(__name__)


def build_health_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["platform"])

    @router.get("/health", response_model=HealthStatus)
    async def health(db: AsyncSession = Depends(get_db_session)) -> HealthStatus:
        db_ok: bool | None = None
        try:
            await db.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("health.db_check_failed", error=str(exc))
            db_ok = False

        return HealthStatus(
            status="ok",
            environment=settings.environment,
            timestamp=datetime.now(timezone.utc),
            db_ok=db_ok,
            redis_ok=await ping_redis(),
            llm_ok=llm_client.is_configured(),
        )

    return router
