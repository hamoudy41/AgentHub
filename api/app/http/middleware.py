from __future__ import annotations

import time
import uuid

import structlog.contextvars
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.core.config import Settings
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.core.redis import check_rate_limit


def install_http_middleware(app: FastAPI, settings: Settings) -> None:
    _install_request_context_middleware(app)
    _install_cors(app, settings)
    _install_security_headers_middleware(app, settings)

    if settings.enable_prometheus:
        _install_metrics_middleware(app)

    if settings.redis_url and settings.api_v1_prefix:
        _install_rate_limit_middleware(app, settings)

    if settings.api_key:
        _install_api_key_middleware(app, settings)


def _install_request_context_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            structlog.contextvars.clear_contextvars()


def _install_cors(app: FastAPI, settings: Settings) -> None:
    origins = (
        [origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()]
        if settings.cors_allowed_origins and settings.cors_allowed_origins.strip() != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _install_security_headers_middleware(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if settings.environment == "prod":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def _install_metrics_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        path = request.scope.get("path", "")
        method = request.method
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
        REQUEST_COUNT.labels(method=method, path=path, status=response.status_code).inc()
        return response


def _install_rate_limit_middleware(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if not request.url.path.startswith(f"{settings.api_v1_prefix}/"):
            return await call_next(request)

        tenant_id = request.headers.get(settings.tenant_header_name) or settings.default_tenant_id
        limit = getattr(settings, "rate_limit_per_minute", 120)
        if not await check_rate_limit(tenant_id, limit=limit, window_seconds=60):
            return ORJSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        return await call_next(request)


def _install_api_key_middleware(app: FastAPI, settings: Settings) -> None:
    exempt_paths = {f"{settings.api_v1_prefix}/health"}

    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if request.url.path in exempt_paths:
            return await call_next(request)

        require_auth = (
            request.url.path.startswith(f"{settings.api_v1_prefix}/")
            or request.url.path == "/metrics"
        )
        if require_auth:
            key = request.headers.get("X-API-Key")
            if key != settings.api_key:
                return ORJSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)
