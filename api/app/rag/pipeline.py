"""RAG pipeline: indexing, retrieval, and query."""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session_factory
from ..models import DocumentChunk
from .chunking import chunk_text
from .embeddings import embedding_service


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGPipeline:
    """Pipeline for indexing documents and retrieving relevant chunks."""

    async def index_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        text: str,
        db: AsyncSession | None = None,
    ) -> int:
        """Index a document: chunk, embed, and store. Returns number of chunks indexed."""
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            return 0

        embeddings: list[list[float]] = []
        for c in chunks:
            vec = await embedding_service.embed(c)
            embeddings.append(vec)

        factory = get_session_factory()

        async def _do(session: AsyncSession) -> int:
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.tenant_id == tenant_id,
                )
            )
            for i, (chunk_text_val, emb) in enumerate(zip(chunks, embeddings)):
                dc = DocumentChunk(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk_text_val,
                    embedding=emb,
                )
                session.add(dc)
            await session.commit()
            return len(chunks)

        if db:
            return await _do(db)
        async with factory() as session:
            return await _do(session)

    async def retrieve(
        self,
        *,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        db: AsyncSession | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top_k chunks most similar to query."""
        query_vec = await embedding_service.embed(query)
        factory = get_session_factory()

        async def _do(session: AsyncSession) -> list[dict[str, Any]]:
            stmt = select(DocumentChunk).where(DocumentChunk.tenant_id == tenant_id)
            if document_ids:
                stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
            result = await session.execute(stmt)
            chunks = result.scalars().all()
            scored = [(c, _cosine_similarity(query_vec, c.embedding)) for c in chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                {
                    "text": c.text,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "score": round(s, 4),
                    "metadata": {"document_id": c.document_id},
                }
                for c, s in scored[:top_k]
            ]

        if db:
            return await _do(db)
        async with factory() as session:
            return await _do(session)

    async def get_chunks(
        self,
        *,
        tenant_id: str,
        document_id: str,
        db: AsyncSession | None = None,
    ) -> list[dict[str, Any]]:
        """Get all chunks for a document (for testing)."""
        factory = get_session_factory()

        async def _do(session: AsyncSession) -> list[dict[str, Any]]:
            stmt = select(DocumentChunk).where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.document_id == document_id,
            )
            result = await session.execute(stmt)
            chunks = result.scalars().all()
            return [{"text": c.text, "chunk_index": c.chunk_index} for c in chunks]

        if db:
            return await _do(db)
        async with factory() as session:
            return await _do(session)


rag_pipeline = RAGPipeline()
