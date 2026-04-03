from typing import AsyncGenerator
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.context import ExecutionContext, clear_execution_context, set_execution_context
from app.db import get_db_session
from app.documents import fetch_document
from app.http.sse import stream_text_tokens
from app.persistence.repositories.document import DocumentRepository
from app.providers.registry import ProviderRegistry
from app.rag.pipeline import rag_pipeline
from app.schemas import RAGIndexRequest, RAGIndexResponse, RAGQueryRequest, RAGQueryResponse
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


def _build_rag_service(db: AsyncSession) -> tuple[RAGService, LLMService]:
    settings = get_settings()
    registry = ProviderRegistry(settings)
    llm_service = LLMService(registry.get_llm_provider())
    rag_service = RAGService(
        llm_service=llm_service,
        embedding_provider=registry.get_embedding_provider(),
        search_provider=registry.get_search_provider(),
        document_repository=DocumentRepository(db),
    )
    return rag_service, llm_service


async def run_rag_query_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: RAGQueryRequest,
) -> dict:
    ctx = ExecutionContext.from_request(tenant_id=tenant_id)
    set_execution_context(ctx)
    rag_service, _ = _build_rag_service(db)
    try:
        docs = await rag_service.retrieve_documents(
            payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
            context=ctx,
        )
        result = await rag_service.answer_question(
            payload.query,
            context_documents=docs,
            context=ctx,
        )
        return {
            "answer": result["answer"],
            "sources": [
                {
                    "text": d["text"],
                    "document_id": d["document_id"],
                    "score": d.get("score", 0.0),
                }
                for d in docs
            ],
            "model": result.get("model", "unknown"),
            "metadata": {"latency_ms": result.get("latency_ms")},
        }
    finally:
        clear_execution_context()


async def run_rag_query_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: RAGQueryRequest,
) -> AsyncGenerator[str, None]:
    ctx = ExecutionContext.from_request(tenant_id=tenant_id)
    set_execution_context(ctx)
    rag_service, llm_service = _build_rag_service(db)
    try:
        docs = await rag_service.retrieve_documents(
            payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
            context=ctx,
        )
        context_text = "\n\n".join(f"[{d['document_id']}] {d['text'][:500]}" for d in docs)
        if not context_text:
            context_text = "(No relevant documents found.)"

        prompt = (
            "Answer the question based only on the following retrieved context. "
            "If the context does not contain the answer, say so briefly.\n\n"
            f"Context:\n{context_text[:8000]}\n\nQuestion: {payload.query}"
        )

        async for token in llm_service.stream_complete(
            prompt,
            system_prompt=(
                "You are a helpful assistant. Answer concisely based only on the given context."
            ),
            context=ctx,
        ):
            yield token
    finally:
        clear_execution_context()


def build_rag_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["rag"])

    @router.post("/ai/rag/query")
    async def rag_query(
        payload: RAGQueryRequest,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> RAGQueryResponse:
        result = await run_rag_query_flow(tenant_id=tenant_id, db=db, payload=payload)
        return RAGQueryResponse(**result)

    @router.post("/ai/rag/query/stream")
    async def rag_query_stream(
        payload: RAGQueryRequest,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> StreamingResponse:
        async def _stream() -> AsyncGenerator[str, None]:
            async for token in run_rag_query_flow_stream(
                tenant_id=tenant_id,
                db=db,
                payload=payload,
            ):
                yield token

        return StreamingResponse(
            stream_text_tokens(
                _stream(),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/ai/rag/index")
    async def rag_index(
        payload: RAGIndexRequest,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
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
