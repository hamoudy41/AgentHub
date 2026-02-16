"""Tests for RAG pipeline (index_document, retrieve, get_chunks)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.rag.pipeline import rag_pipeline


@pytest.mark.asyncio
async def test_pipeline_index_document_with_db(db_session):
    """index_document chunks, embeds, and stores chunks."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )
        n = await rag_pipeline.index_document(
            tenant_id="t1",
            document_id="doc-1",
            text="First chunk. Second chunk. Third chunk.",
            db=db_session,
        )
        assert n >= 1

    chunks = await rag_pipeline.get_chunks(tenant_id="t1", document_id="doc-1", db=db_session)
    assert len(chunks) >= 1
    assert all("text" in c and "chunk_index" in c for c in chunks)


@pytest.mark.asyncio
async def test_pipeline_index_document_empty_text_returns_zero(db_session):
    """index_document returns 0 for empty text."""
    n = await rag_pipeline.index_document(
        tenant_id="t1", document_id="doc-empty", text="", db=db_session
    )
    assert n == 0


@pytest.mark.asyncio
async def test_pipeline_retrieve_returns_top_k(db_session):
    """retrieve returns top_k chunks by similarity."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )
        await rag_pipeline.index_document(
            tenant_id="t1",
            document_id="doc-a",
            text="Paris is the capital of France. It has the Eiffel Tower.",
            db=db_session,
        )

    results = await rag_pipeline.retrieve(
        tenant_id="t1", query="What is the capital of France?", top_k=2, db=db_session
    )
    assert len(results) <= 2
    for r in results:
        assert "text" in r and "document_id" in r and "score" in r and "metadata" in r


@pytest.mark.asyncio
async def test_pipeline_retrieve_with_document_ids_filter(db_session):
    """retrieve filters by document_ids when provided."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )
        await rag_pipeline.index_document(
            tenant_id="t1",
            document_id="doc-x",
            text="Content X.",
            db=db_session,
        )
        await rag_pipeline.index_document(
            tenant_id="t1",
            document_id="doc-y",
            text="Content Y.",
            db=db_session,
        )

    results = await rag_pipeline.retrieve(
        tenant_id="t1",
        query="Content",
        top_k=5,
        document_ids=["doc-x"],
        db=db_session,
    )
    assert all(r["document_id"] == "doc-x" for r in results)


@pytest.mark.asyncio
async def test_pipeline_index_document_without_db(db_session):
    """index_document works without passing db (creates own session)."""
    from types import SimpleNamespace

    with patch("app.rag.embeddings.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            embedding_model="mock", embedding_dimension=384
        )
        n = await rag_pipeline.index_document(
            tenant_id="t1",
            document_id="doc-standalone",
            text="Standalone content.",
        )
        assert n >= 1
