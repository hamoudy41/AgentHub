"""RAG query flows."""

from __future__ import annotations

from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from .rag.pipeline import rag_pipeline
from .schemas import RAGQueryRequest
from .services_llm import LLMError, llm_client


async def run_rag_query_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: RAGQueryRequest,
) -> dict[str, Any]:
    """Run RAG query: retrieve chunks, build context, call LLM."""
    chunks = await rag_pipeline.retrieve(
        tenant_id=tenant_id,
        query=payload.query,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        db=db,
    )
    context = "\n\n".join(f"[{c['document_id']}] {c['text']}" for c in chunks)
    if not context:
        context = "(No relevant documents found.)"

    prompt = (
        "Answer the question based only on the following retrieved context. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{context[:8000]}\n\nQuestion: {payload.query}"
    )

    if not llm_client.is_configured():
        return {
            "answer": "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL.",
            "sources": [
                {"text": c["text"], "document_id": c["document_id"], "score": c["score"]}
                for c in chunks
            ],
            "model": "fallback",
            "metadata": {"error": "llm_not_configured"},
        }

    try:
        result = await llm_client.complete(
            prompt,
            system_prompt="You are a helpful assistant. Answer concisely based only on the given context.",
            tenant_id=tenant_id,
        )
        return {
            "answer": result.raw_text,
            "sources": [
                {"text": c["text"], "document_id": c["document_id"], "score": c["score"]}
                for c in chunks
            ],
            "model": result.model,
            "metadata": {"latency_ms": result.latency_ms},
        }
    except (LLMError, Exception):
        return {
            "answer": "Answer unavailable (model error).",
            "sources": [
                {"text": c["text"], "document_id": c["document_id"], "score": c["score"]}
                for c in chunks
            ],
            "model": "fallback",
            "metadata": {"error": "llm_error"},
        }


async def run_rag_query_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: RAGQueryRequest,
) -> AsyncIterator[str]:
    """Stream RAG query response from LLM."""
    chunks = await rag_pipeline.retrieve(
        tenant_id=tenant_id,
        query=payload.query,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        db=db,
    )
    context = "\n\n".join(f"[{c['document_id']}] {c['text']}" for c in chunks)
    if not context:
        context = "(No relevant documents found.)"

    prompt = (
        "Answer the question based only on the following retrieved context. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{context[:8000]}\n\nQuestion: {payload.query}"
    )

    if not llm_client.is_configured():
        yield "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL."
        return

    try:
        async for token in llm_client.stream_complete(
            prompt,
            system_prompt="You are a helpful assistant. Answer concisely based only on the given context.",
            tenant_id=tenant_id,
        ):
            yield token
    except (LLMError, Exception):
        yield "Answer unavailable (model error)."
