import orjson
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context import ExecutionContext, clear_execution_context, set_execution_context
from app.core.errors import ConflictError, NotFoundError
from app.core.redis import cache_key, get_cached, set_cached
from app.db import get_db_session
from app.documents import (
    UploadTooLargeError,
    UploadValidationError,
    prepare_uploaded_document,
)
from app.persistence.repositories.document import DocumentRepository
from app.schemas import DocumentCreate, DocumentRead
from app.services.document_service import DocumentService


def _build_document_service(db: AsyncSession) -> DocumentService:
    return DocumentService(DocumentRepository(db))


def build_documents_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["documents"])

    @router.post("/documents", status_code=status.HTTP_201_CREATED)
    async def create_document_route(
        payload: DocumentCreate,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> DocumentRead:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        service = _build_document_service(db)
        try:
            return await service.create(
                document_id=payload.id,
                title=payload.title,
                text=payload.text,
                context=ctx,
            )
        except ConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document with ID '{exc}' already exists. "
                    "Use Get by ID to view, or choose a different ID."
                ),
            ) from exc
        finally:
            clear_execution_context()

    @router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
    async def upload_document(
        file: Annotated[UploadFile, File(...)],
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
        document_id: Annotated[Optional[str], Form()] = None,
        title: Annotated[Optional[str], Form()] = None,
    ) -> DocumentRead:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        service = _build_document_service(db)
        try:
            prepared = await prepare_uploaded_document(
                file,
                document_id=document_id,
                title=title,
            )
            return await service.create(
                document_id=prepared.document_id,
                title=prepared.title,
                text=prepared.text,
                context=ctx,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
            ) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document with ID '{exc}' already exists. "
                    "Use a different ID or Get by ID to view."
                ),
            ) from exc
        finally:
            clear_execution_context()

    @router.get("/documents/{document_id}")
    async def get_document_route(
        document_id: str,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> DocumentRead:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        try:
            service = _build_document_service(db)
            cache_entry = await get_cached(cache_key(tenant_id, "document", document_id))
            if cache_entry:
                try:
                    return DocumentRead.model_validate(orjson.loads(cache_entry))
                except (TypeError, ValueError):
                    pass

            response = await service.read(document_id, context=ctx)

            await set_cached(
                cache_key(tenant_id, "document", document_id),
                orjson.dumps(response.model_dump(mode="json")).decode(),
                ttl_seconds=300,
            )
            return response
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            ) from exc
        finally:
            clear_execution_context()

    return router
