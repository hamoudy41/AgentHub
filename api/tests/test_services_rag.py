"""Tests for RAG query flows (run_rag_query_flow, run_rag_query_flow_stream)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import RAGQueryRequest
from app.services_rag import run_rag_query_flow, run_rag_query_flow_stream


def _llm_result(raw_text: str, model: str = "llama", latency_ms: float = 10.0):
    return type("LLMResult", (), {"raw_text": raw_text, "model": model, "latency_ms": latency_ms})()


@pytest.mark.asyncio
async def test_rag_query_flow_llm_configured_success(db_session):
    """run_rag_query_flow returns answer when LLM is configured."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(
            return_value=[{"text": "Paris is capital.", "document_id": "d1", "score": 0.9}]
        )
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = True
            mock_llm.complete = AsyncMock(return_value=_llm_result("Paris.", "llama"))
            out = await run_rag_query_flow(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Capital?", top_k=5),
            )
            assert out["answer"] == "Paris."
            assert out["model"] == "llama"
            assert len(out["sources"]) == 1
            assert out["sources"][0]["document_id"] == "d1"


@pytest.mark.asyncio
async def test_rag_query_flow_llm_not_configured_returns_fallback(db_session):
    """run_rag_query_flow returns fallback when LLM not configured."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(
            return_value=[{"text": "X", "document_id": "d1", "score": 0.8}]
        )
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = False
            out = await run_rag_query_flow(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Q?", top_k=5),
            )
            assert "LLM not configured" in out["answer"]
            assert out["model"] == "fallback"
            assert "llm_not_configured" in out["metadata"].get("error", "")


@pytest.mark.asyncio
async def test_rag_query_flow_llm_error_returns_fallback(db_session):
    """run_rag_query_flow returns fallback on LLM error."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(return_value=[])
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = True
            mock_llm.complete = AsyncMock(side_effect=Exception("API down"))
            out = await run_rag_query_flow(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Q?", top_k=5),
            )
            assert "unavailable" in out["answer"].lower() or "error" in out["answer"].lower()
            assert out["model"] == "fallback"


@pytest.mark.asyncio
async def test_rag_query_flow_stream_llm_configured(db_session):
    """run_rag_query_flow_stream yields tokens when LLM configured."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(return_value=[])
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = True

            async def fake_stream(*args, **kwargs):
                yield "Paris"
                yield " is capital."

            mock_llm.stream_complete = fake_stream
            tokens = []
            async for t in run_rag_query_flow_stream(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Capital?", top_k=5),
            ):
                tokens.append(t)
            assert "".join(tokens) == "Paris is capital."


@pytest.mark.asyncio
async def test_rag_query_flow_stream_llm_not_configured(db_session):
    """run_rag_query_flow_stream yields fallback when LLM not configured."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(return_value=[])
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = False
            tokens = []
            async for t in run_rag_query_flow_stream(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Q?", top_k=5),
            ):
                tokens.append(t)
            assert "".join(tokens) == "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL."


@pytest.mark.asyncio
async def test_rag_query_flow_stream_llm_error(db_session):
    """run_rag_query_flow_stream yields fallback on LLM error."""
    with patch("app.services_rag.rag_pipeline") as mock_pipeline:
        mock_pipeline.retrieve = AsyncMock(return_value=[])
        with patch("app.services_rag.llm_client") as mock_llm:
            mock_llm.is_configured.return_value = True
            mock_llm.stream_complete = AsyncMock(side_effect=Exception("Stream failed"))
            tokens = []
            async for t in run_rag_query_flow_stream(
                tenant_id="t1",
                db=db_session,
                payload=RAGQueryRequest(query="Q?", top_k=5),
            ):
                tokens.append(t)
            assert "unavailable" in "".join(tokens).lower() or "error" in "".join(tokens).lower()
