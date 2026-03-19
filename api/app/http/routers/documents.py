from __future__ import annotations

import orjson

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import cache_key, get_cached, set_cached
from app.db import get_db_session
from app.documents import (
    DocumentConflictError,
    UploadTooLargeError,
    UploadValidationError,
    create_document,
    document_to_read,
    fetch_document,
    prepare_uploaded_document,
)
from app.schemas import DocumentCreate, DocumentRead


def build_documents_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["documents"])

    @router.post("/documents", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
    async def create_document_route(
        payload: DocumentCreate,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        try:
            document = await create_document(
                db,
                tenant_id,
                document_id=payload.id,
                title=payload.title,
                text=payload.text,
            )
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document with ID '{exc}' already exists. "
                    "Use Get by ID to view, or choose a different ID."
                ),
            ) from exc
        return document_to_read(document)

    @router.post(
        "/documents/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED
    )
    async def upload_document(
        file: UploadFile = File(...),
        document_id: str | None = Form(None),
        title: str | None = Form(None),
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        try:
            prepared = await prepare_uploaded_document(
                file,
                document_id=document_id,
                title=title,
            )
            document = await create_document(
                db,
                tenant_id,
                document_id=prepared.document_id,
                title=prepared.title,
                text=prepared.text,
            )
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)
            ) from exc
        except UploadValidationError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Document with ID '{exc}' already exists. "
                    "Use a different ID or Get by ID to view."
                ),
            ) from exc
        return document_to_read(document)

    @router.get("/documents/{document_id}", response_model=DocumentRead)
    async def get_document_route(
        document_id: str,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> DocumentRead:
        cache_entry = await get_cached(cache_key(tenant_id, "document", document_id))
        if cache_entry:
            try:
                return DocumentRead.model_validate(orjson.loads(cache_entry))
            except (orjson.JSONDecodeError, TypeError, ValueError):
                pass

        document = await fetch_document(db, tenant_id, document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        response = document_to_read(document)
        await set_cached(
            cache_key(tenant_id, "document", document_id),
            orjson.dumps(response.model_dump(mode="json")).decode(),
            ttl_seconds=300,
        )
        return response

    return router
