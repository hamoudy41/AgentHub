from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.documents import fetch_document
from app.http.sse import stream_text_tokens
from app.rag.pipeline import rag_pipeline
from app.schemas import RAGIndexRequest, RAGIndexResponse, RAGQueryRequest, RAGQueryResponse
from app.services_rag import run_rag_query_flow, run_rag_query_flow_stream


def build_rag_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["rag"])

    @router.post("/ai/rag/query", response_model=RAGQueryResponse)
    async def rag_query(
        payload: RAGQueryRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> RAGQueryResponse:
        result = await run_rag_query_flow(tenant_id=tenant_id, db=db, payload=payload)
        return RAGQueryResponse(**result)

    @router.post("/ai/rag/query/stream")
    async def rag_query_stream(
        payload: RAGQueryRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> StreamingResponse:
        return StreamingResponse(
            stream_text_tokens(
                run_rag_query_flow_stream(tenant_id=tenant_id, db=db, payload=payload),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/ai/rag/index", response_model=RAGIndexResponse)
    async def rag_index(
        payload: RAGIndexRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> RAGIndexResponse:
        document = await fetch_document(db, tenant_id, payload.document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{payload.document_id}' not found",
            )

        chunks_indexed = await rag_pipeline.index_document(
            tenant_id=tenant_id,
            document_id=payload.document_id,
            text=document.text,
            db=db,
        )
        return RAGIndexResponse(document_id=payload.document_id, chunks_indexed=chunks_indexed)

    return router
