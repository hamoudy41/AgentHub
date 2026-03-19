from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import ORJSONResponse

from app.core.logging import get_logger
from app.services_ai_flows import AiFlowError


logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AiFlowError)
    async def ai_flow_error_handler(_: Request, exc: AiFlowError) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "error_type": "ai_flow_error"},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> ORJSONResponse:
        if isinstance(exc, HTTPException):
            return ORJSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("app.unhandled_error", error=str(exc), exc_info=True)
        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_type": "internal_error"},
        )
