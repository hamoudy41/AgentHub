"""Tests for RAG pipeline and API (TDD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# --- Chunking tests ---


def test_chunk_text_splits_by_size():
    """Chunking splits text into chunks of configured size with overlap."""
    from app.rag.chunking import chunk_text

    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 250 for c in chunks)
    # All content should appear (possibly with overlap)
    combined = "".join(chunks)
    assert "A" * 100 in combined


def test_chunk_text_respects_sentence_boundaries():
    """Chunking prefers to split at sentence boundaries when possible."""
    from app.rag.chunking import chunk_text

    text = "First sentence. Second sentence. Third sentence. Fourth."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=5)
    assert len(chunks) >= 2
    assert any(c.strip().endswith(".") for c in chunks)


# --- Embedding service tests ---


@pytest.mark.asyncio
async def test_embedding_service_returns_fixed_dimension():
    """Embedding service returns vectors of consistent dimension."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )

        from app.rag.embeddings import embedding_service

        vec = await embedding_service.embed("Hello world")
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)


@pytest.mark.asyncio
async def test_embedding_service_same_text_same_vector():
    """Embedding service returns same vector for same text (deterministic mock)."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )

        from app.rag.embeddings import embedding_service

        v1 = await embedding_service.embed("Hello")
        v2 = await embedding_service.embed("Hello")
        assert v1 == v2


# --- API tests ---


@pytest.mark.asyncio
async def test_rag_query_endpoint_returns_200(client, tenant_headers):
    """POST /ai/rag/query returns 200 with answer."""
    with patch("app.api.run_rag_query_flow", new_callable=AsyncMock) as mock_flow:
        mock_flow.return_value = {"answer": "Paris", "sources": [], "model": "llama3.2"}

        r = await client.post(
            "/api/v1/ai/rag/query",
            headers=tenant_headers,
            json={"query": "What is the capital of France?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["answer"] == "Paris"


@pytest.mark.asyncio
async def test_rag_query_endpoint_validates_payload(client, tenant_headers):
    """POST /ai/rag/query returns 422 for empty query."""
    r = await client.post(
        "/api/v1/ai/rag/query",
        headers=tenant_headers,
        json={"query": ""},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_rag_query_endpoint_accepts_document_ids_filter(client, tenant_headers):
    """POST /ai/rag/query accepts optional document_ids to scope search."""
    with patch("app.api.run_rag_query_flow", new_callable=AsyncMock) as mock_flow:
        mock_flow.return_value = {"answer": "From doc1", "sources": [], "model": "llama3.2"}

        r = await client.post(
            "/api/v1/ai/rag/query",
            headers=tenant_headers,
            json={"query": "Summarize", "document_ids": ["doc-1", "doc-2"]},
        )
        assert r.status_code == 200
        mock_flow.assert_called_once()
        call_payload = mock_flow.call_args.kwargs["payload"]
        assert call_payload.document_ids == ["doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_rag_index_endpoint_indexes_document(client, tenant_headers):
    """POST /ai/rag/index indexes a document for RAG search."""
    await client.post(
        "/api/v1/documents",
        headers=tenant_headers,
        json={"id": "rag-doc", "title": "RAG Doc", "text": "Content for RAG indexing."},
    )
    with patch("app.api.rag_pipeline") as mock_pipeline:
        mock_pipeline.index_document = AsyncMock(return_value=1)

        r = await client.post(
            "/api/v1/ai/rag/index",
            headers=tenant_headers,
            json={"document_id": "rag-doc"},
        )
        assert r.status_code in (200, 202)
        mock_pipeline.index_document.assert_called_once()


@pytest.mark.asyncio
async def test_rag_query_stream_endpoint_returns_sse(client, tenant_headers):
    """POST /ai/rag/query/stream returns SSE streaming response."""
    with patch("app.api.run_rag_query_flow_stream") as mock_stream:

        async def fake_stream(*args, **kwargs):
            yield "Paris"
            yield " is the capital."

        mock_stream.return_value = fake_stream()

        async with client.stream(
            "POST",
            "/api/v1/ai/rag/query/stream",
            headers=tenant_headers,
            json={"query": "What is the capital of France?"},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
