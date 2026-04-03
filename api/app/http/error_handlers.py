from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import ORJSONResponse

from app.core.errors import AppError
from app.core.logging import get_logger


logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "detail": str(exc),
                "error_type": exc.error_code,
                "details": exc.details,
            },
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
