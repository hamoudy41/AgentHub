from __future__ import annotations

from datetime import datetime, timezone

import orjson
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .core.config import get_settings
from .core.logging import configure_logging, get_logger
from .core.metrics import REQUEST_COUNT, REQUEST_LATENCY, get_metrics, metrics_content_type
from .core.redis import cache_key, check_rate_limit, close_redis, get_cached, set_cached
from .db import get_db_session, get_engine
from .models import Base, Document
from .schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    DocumentCreate,
    DocumentRead,
    HealthStatus,
    NotarySummarizeRequest,
    NotarySummarizeResponse,
)
from .services_ai_flows import (
    AiFlowError,
    run_ask_flow,
    run_classify_flow,
    run_notary_summarization_flow,
)
from .services_llm import llm_client


logger = get_logger(__name__)

# Deterministic fallback for legacy documents with NULL created_at (avoids mutable timestamps).
_LEGACY_CREATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Document model column limits (String(64), String(255)).
_MAX_DOCUMENT_ID_LEN = 64
_MAX_DOCUMENT_TITLE_LEN = 255


def _is_duplicate_key_error(exc: BaseException) -> bool:
    """Detect duplicate/unique constraint violations only; not FK, CHECK, or NOT NULL."""
    if not isinstance(exc, IntegrityError):
        return _msg_indicates_duplicate(f"{exc}")
    orig = getattr(exc, "orig", None)
    if orig is not None:
        # PostgreSQL: sqlstate 23505 = unique_violation (psycopg, asyncpg)
        sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if sqlstate == "23505":
            return True
        # asyncpg: UniqueViolationError (avoids coupling via try/import)
        if type(orig).__name__ == "UniqueViolationError":
            return True
        cause = getattr(orig, "__cause__", None)
        if cause is not None and type(cause).__name__ == "UniqueViolationError":
            return True
    return _msg_indicates_duplicate(f"{exc}")


def _msg_indicates_duplicate(msg: str) -> bool:
    """Parse error message for duplicate/unique hints; exclude FK, CHECK, NOT NULL."""
    lower = msg.lower()
    if any(k in lower for k in ("foreign key", "check constraint", "not null", "notnull")):
        return False
    return any(
        k in lower
        for k in (
            "unique constraint",
            "unique violation",
            "duplicate key",
            "duplicate entry",
            "already exists",
        )
    )


async def _init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_app() -> FastAPI:
    configure_logging()

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await _init_db()
        logger.info("app.startup")
        yield
        await close_redis()
        logger.info("app.shutdown")

    app.router.lifespan_context = lifespan

    if settings.enable_prometheus:

        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            import time

            start = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            path = request.scope.get("path", "")
            method = request.method
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            REQUEST_COUNT.labels(method=method, path=path, status=response.status_code).inc()
            return response

    if settings.redis_url and settings.api_v1_prefix:

        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            if not request.url.path.startswith(f"{settings.api_v1_prefix}/"):
                return await call_next(request)
            tenant_id = (
                request.headers.get(settings.tenant_header_name) or settings.default_tenant_id
            )
            limit = getattr(settings, "rate_limit_per_minute", 120)
            if not await check_rate_limit(tenant_id, limit=limit, window_seconds=60):
                return ORJSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
            return await call_next(request)

    if settings.api_key:
        _no_auth_paths = {"/metrics", f"{settings.api_v1_prefix}/health"}

        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            if request.url.path in _no_auth_paths:
                return await call_next(request)
            if request.url.path.startswith(f"{settings.api_v1_prefix}/"):
                key = request.headers.get("X-API-Key")
                if key != settings.api_key:
                    return ORJSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid or missing API key"},
                    )
            return await call_next(request)

    if settings.enable_prometheus:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            return Response(content=get_metrics(), media_type=metrics_content_type())

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

    api_router = APIRouter(prefix=settings.api_v1_prefix)

    async def get_tenant_id(
        x_tenant_id: str | None = Header(default=None, alias=settings.tenant_header_name),
    ) -> str:
        return x_tenant_id or settings.default_tenant_id

    @api_router.get("/health", response_model=HealthStatus)
    async def health(db: AsyncSession = Depends(get_db_session)) -> HealthStatus:
        db_ok: bool | None = None
        try:
            await db.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("health.db_check_failed", error=str(exc))
            db_ok = False
        llm_ok = llm_client.is_configured()
        return HealthStatus(
            environment=settings.environment,
            timestamp=datetime.now(timezone.utc),
            db_ok=db_ok,
            llm_ok=llm_ok,
        )

    @api_router.post("/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
    async def create_document(
        payload: DocumentCreate,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        result = await db.execute(
            select(Document).where(
                Document.id == payload.id,
                Document.tenant_id == tenant_id,
            )
        )
        if result.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with ID '{payload.id}' already exists. Use Get by ID to view, or choose a different ID.",
            )
        doc = Document(
            id=payload.id,
            tenant_id=tenant_id,
            title=payload.title,
            text=payload.text,
        )
        db.add(doc)
        try:
            await db.commit()
            await db.refresh(doc)
        except Exception as e:
            await db.rollback()
            if _is_duplicate_key_error(e):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document with ID '{payload.id}' already exists. Use Get by ID to view, or choose a different ID.",
                )
            raise
        return _doc_to_read(doc)

    @api_router.post(
        "/documents/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED
    )
    async def upload_document(
        file: UploadFile = File(...),
        document_id: str | None = Form(None),
        title: str | None = Form(None),
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        _max_size = 5 * 1024 * 1024  # 5 MB
        body = await file.read()
        if len(body) > _max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {_max_size // 1024 // 1024} MB)",
            )
        try:
            text_content = body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File could not be decoded as UTF-8 text",
            )
        doc_id = document_id or (Path(file.filename or "upload").stem or "upload")
        doc_title = title or (file.filename or "Uploaded document")
        if len(doc_id) > _MAX_DOCUMENT_ID_LEN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document ID must be at most {_MAX_DOCUMENT_ID_LEN} characters. "
                f"Provide a shorter document_id or use a shorter filename.",
            )
        doc_title = doc_title[:_MAX_DOCUMENT_TITLE_LEN]
        result = await db.execute(
            select(Document).where(
                Document.id == doc_id,
                Document.tenant_id == tenant_id,
            )
        )
        if result.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with ID '{doc_id}' already exists. Use a different ID or Get by ID to view.",
            )
        doc = Document(
            id=doc_id,
            tenant_id=tenant_id,
            title=doc_title,
            text=text_content,
        )
        db.add(doc)
        try:
            await db.commit()
            await db.refresh(doc)
        except Exception as e:
            await db.rollback()
            if _is_duplicate_key_error(e):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Document with ID '{doc_id}' already exists. Use a different ID or Get by ID to view.",
                )
            raise
        return _doc_to_read(doc)

    def _doc_to_read(doc: Document) -> DocumentRead:
        """Build DocumentRead, handling None created_at for legacy documents."""
        created = doc.created_at if doc.created_at is not None else _LEGACY_CREATED_AT
        return DocumentRead(
            id=doc.id,
            title=doc.title,
            text=doc.text,
            created_at=created,
        )

    @api_router.get("/documents/{document_id}", response_model=DocumentRead)
    async def get_document(
        document_id: str,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        ck = cache_key(tenant_id, "document", document_id)
        cached = await get_cached(ck)
        if cached:
            try:
                return DocumentRead.model_validate(orjson.loads(cached))
            except (orjson.JSONDecodeError, ValueError, TypeError, Exception):
                pass
        doc = await db.get(Document, document_id)
        if not doc or doc.tenant_id != tenant_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        out = _doc_to_read(doc)
        await set_cached(ck, orjson.dumps(out.model_dump(mode="json")).decode(), ttl_seconds=300)
        return out

    @api_router.post(
        "/ai/notary/summarize",
        response_model=NotarySummarizeResponse,
        status_code=status.HTTP_200_OK,
    )
    async def notary_summarize(
        payload: NotarySummarizeRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> NotarySummarizeResponse:
        return await run_notary_summarization_flow(
            tenant_id=tenant_id,
            db=db,
            payload=payload,
        )

    @api_router.post(
        "/ai/classify",
        response_model=ClassifyResponse,
        status_code=status.HTTP_200_OK,
    )
    async def classify(
        payload: ClassifyRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> ClassifyResponse:
        return await run_classify_flow(tenant_id=tenant_id, db=db, payload=payload)

    @api_router.post(
        "/ai/ask",
        response_model=AskResponse,
        status_code=status.HTTP_200_OK,
    )
    async def ask(
        payload: AskRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> AskResponse:
        return await run_ask_flow(tenant_id=tenant_id, db=db, payload=payload)

    app.include_router(api_router)

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app


app = create_app()
