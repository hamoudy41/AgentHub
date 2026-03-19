from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.metrics import get_metrics, metrics_content_type
from app.core.redis import close_redis
from app.db import get_engine
from app.http.dependencies import build_tenant_dependency
from app.http.error_handlers import register_error_handlers
from app.http.middleware import install_http_middleware
from app.http.routers.agents import build_agent_router
from app.http.routers.documents import build_documents_router
from app.http.routers.health import build_health_router
from app.http.routers.rag import build_rag_router
from app.http.routers.workflows import build_workflow_router
from app.models import Base


logger = get_logger(__name__)


async def _init_db() -> None:
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


def _build_lifespan():
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await _init_db()
        logger.info("app.startup")
        yield
        await close_redis()
        logger.info("app.shutdown")

    return lifespan


def _register_metrics_endpoint(app: FastAPI) -> None:
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=get_metrics(), media_type=metrics_content_type())


def _register_api_routes(app: FastAPI) -> None:
    settings = get_settings()
    tenant_dependency = build_tenant_dependency(settings)
    api_router = APIRouter(prefix=settings.api_v1_prefix)
    api_router.include_router(build_health_router(settings))
    api_router.include_router(build_documents_router(tenant_dependency))
    api_router.include_router(build_workflow_router(tenant_dependency))
    api_router.include_router(build_rag_router(tenant_dependency))
    api_router.include_router(build_agent_router(tenant_dependency))
    app.include_router(api_router)


def _mount_static_frontend(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )
    app.router.lifespan_context = _build_lifespan()

    install_http_middleware(app, settings)
    register_error_handlers(app)

    if settings.enable_prometheus:
        _register_metrics_endpoint(app)

    _register_api_routes(app)
    _mount_static_frontend(app)
    return app


app = create_app()
