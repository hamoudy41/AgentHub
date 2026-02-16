from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.db import get_engine, get_session_factory
from app.models import Document
from app.schemas import (
    AskRequest,
    ClassifyRequest,
    NotarySummarizeRequest,
)
from app.services_ai_flows import (
    run_ask_flow,
    run_ask_flow_stream,
    run_classify_flow,
    run_notary_summarization_flow,
)
from app.services_llm import LLMError, LLMResult


@pytest.fixture
async def db_session():
    from app.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = get_session_factory()
    async with factory() as session:
        yield session


@pytest.mark.asyncio
async def test_notary_flow_uses_document_text_when_document_id_given(db_session):
    doc_id = f"flow-doc-{uuid.uuid4().hex[:8]}"
    doc = Document(id=doc_id, tenant_id="tenant-1", title="D", text="Stored document content.")
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)

    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.generate_notary_summary = AsyncMock(
            return_value=LLMResult(raw_text="Summary of stored doc.", model="mock", latency_ms=1.0),
        )
        out = await run_notary_summarization_flow(
            tenant_id="tenant-1",
            db=db_session,
            payload=NotarySummarizeRequest(document_id=doc_id, text="ignored", language="en"),
        )
        call_args = mock_llm.generate_notary_summary.call_args
        prompt = call_args[0][0]
        assert "Stored document content" in prompt
        assert "ignored" not in prompt or "Stored document content" in prompt
        assert out.summary.raw_summary == "Summary of stored doc."


@pytest.mark.asyncio
async def test_notary_flow_uses_payload_text_when_document_id_missing(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.generate_notary_summary = AsyncMock(
            return_value=LLMResult(raw_text="Summary.", model="mock", latency_ms=1.0),
        )
        out = await run_notary_summarization_flow(
            tenant_id="t1",
            db=db_session,
            payload=NotarySummarizeRequest(text="Inline text here.", language="nl"),
        )
        call_args = mock_llm.generate_notary_summary.call_args
        assert "Inline text here" in call_args[0][0]
        assert out.source == "llm"


@pytest.mark.asyncio
async def test_notary_flow_fallback_when_llm_raises(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.generate_notary_summary = AsyncMock(side_effect=LLMError("Unavailable"))
        out = await run_notary_summarization_flow(
            tenant_id="t1",
            db=db_session,
            payload=NotarySummarizeRequest(text="Some text.", language="en"),
        )
        assert out.source == "fallback"
        assert (
            "Automatische samenvatting" in out.summary.raw_summary
            or "niet beschikbaar" in out.summary.raw_summary
        )


@pytest.mark.asyncio
async def test_notary_document_id_but_doc_not_found_uses_payload_text(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.generate_notary_summary = AsyncMock(
            return_value=LLMResult(raw_text="Summary.", model="mock", latency_ms=1.0),
        )
        await run_notary_summarization_flow(
            tenant_id="tenant-1",
            db=db_session,
            payload=NotarySummarizeRequest(
                document_id="nonexistent-id", text="Fallback text.", language="en"
            ),
        )
        call_args = mock_llm.generate_notary_summary.call_args
        assert "Fallback text" in call_args[0][0]


@pytest.mark.asyncio
async def test_classify_label_matching(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.complete = AsyncMock(
            return_value=LLMResult(raw_text="  invoice  ", model="mock", latency_ms=1.0),
        )
        out = await run_classify_flow(
            tenant_id="t1",
            db=db_session,
            payload=ClassifyRequest(
                text="Invoice for 100 EUR.", candidate_labels=["contract", "invoice", "letter"]
            ),
        )
        assert out.label == "invoice"
        assert out.source == "llm"
        assert out.confidence == 0.9


@pytest.mark.asyncio
async def test_classify_label_not_in_candidates_uses_first(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.complete = AsyncMock(
            return_value=LLMResult(raw_text="  unknown  ", model="mock", latency_ms=1.0),
        )
        out = await run_classify_flow(
            tenant_id="t1",
            db=db_session,
            payload=ClassifyRequest(text="Something.", candidate_labels=["contract", "invoice"]),
        )
        assert out.label == "contract"
        assert out.source == "llm"


@pytest.mark.asyncio
async def test_classify_fallback_on_exception(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("Model down"))
        out = await run_classify_flow(
            tenant_id="t1",
            db=db_session,
            payload=ClassifyRequest(text="X", candidate_labels=["a", "b", "c"]),
        )
        assert out.source == "fallback"
        assert out.label == "a"
        assert out.confidence == 0.0


@pytest.mark.asyncio
async def test_ask_success(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.complete = AsyncMock(
            return_value=LLMResult(raw_text="The total is 50 EUR.", model="mock", latency_ms=1.0),
        )
        out = await run_ask_flow(
            tenant_id="t1",
            db=db_session,
            payload=AskRequest(question="What is the total?", context="Total: 50 EUR."),
        )
        assert out.answer == "The total is 50 EUR."
        assert out.source == "llm"


@pytest.mark.asyncio
async def test_ask_fallback_on_exception(db_session):
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.complete = AsyncMock(side_effect=LLMError("Timeout"))
        out = await run_ask_flow(
            tenant_id="t1",
            db=db_session,
            payload=AskRequest(question="Q?", context="C"),
        )
        assert out.source == "fallback"
        assert "unavailable" in out.answer.lower() or "error" in out.answer.lower()


@pytest.mark.asyncio
async def test_ask_stream_success(db_session):
    """run_ask_flow_stream yields tokens from llm_client.stream_complete."""
    with patch("app.services_ai_flows.llm_client") as mock_llm:

        async def fake_stream(*args, **kwargs):
            yield "The "
            yield "answer."

        mock_llm.is_configured.return_value = True
        mock_llm.stream_complete = fake_stream
        tokens = []
        async for t in run_ask_flow_stream(
            tenant_id="t1",
            db=db_session,
            payload=AskRequest(question="Q?", context="C"),
        ):
            tokens.append(t)
        assert "".join(tokens) == "The answer."


@pytest.mark.asyncio
async def test_ask_stream_fallback_on_exception(db_session):
    """run_ask_flow_stream yields fallback message when stream_complete raises."""
    with patch("app.services_ai_flows.llm_client") as mock_llm:
        mock_llm.is_configured.return_value = True
        mock_llm.stream_complete = AsyncMock(side_effect=RuntimeError("Stream failed"))
        tokens = []
        async for t in run_ask_flow_stream(
            tenant_id="t1",
            db=db_session,
            payload=AskRequest(question="Q?", context="C"),
        ):
            tokens.append(t)
        assert "".join(tokens) == "Answer unavailable (model error)."
