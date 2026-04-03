from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.core.errors import AppError
from app.http.routers import rag as rag_router
from app.http.routers import workflows as wf_router
from app.schemas import AskRequest, ClassifyRequest, NotarySummarizeRequest, RAGQueryRequest


def _tenant_getter() -> str:
    return "tenant-1"


def _raise_value_error(*_args, **_kwargs):
    raise ValueError("bad")


def _raise_sanitized_error(*_args, **_kwargs):
    raise ValueError("sanitized")


def _raise_app_error(message: str, status_code: int):
    raise AppError(message, status_code=status_code)


def _raise_runtime_error(*_args, **_kwargs):
    raise RuntimeError("boom")


def test_workflow_validators_error_paths(monkeypatch):
    with pytest.raises(Exception):
        wf_router._validate_ask_request(AskRequest(question=" ", context="ctx"), "tenant-1")

    with pytest.raises(Exception):
        wf_router._validate_ask_request(AskRequest(question="q", context=" "), "tenant-1")

    with pytest.raises(Exception):
        wf_router._validate_classify_request(
            ClassifyRequest(text=" ", candidate_labels=["a"]), "tenant-1"
        )

    with pytest.raises(Exception):
        wf_router._validate_classify_request(
            ClassifyRequest(text="text", candidate_labels=[]), "tenant-1"
        )

    with pytest.raises(Exception):
        wf_router._validate_notary_request(
            NotarySummarizeRequest(text=" ", language="nl"), "tenant-1"
        )

    monkeypatch.setattr("app.http.routers.workflows.sanitize_user_input", _raise_value_error)
    with pytest.raises(Exception):
        wf_router._validate_notary_request(
            NotarySummarizeRequest(text="x", language="nl"), "tenant-1"
        )


def test_workflow_validators_convert_sanitize_value_errors(monkeypatch):
    monkeypatch.setattr(
        "app.http.routers.workflows.sanitize_user_input",
        _raise_sanitized_error,
    )

    with pytest.raises(Exception):
        wf_router._validate_ask_request(AskRequest(question="q", context="ctx"), "tenant-1")

    with pytest.raises(Exception):
        wf_router._validate_classify_request(
            ClassifyRequest(text="text", candidate_labels=["a"]),
            "tenant-1",
        )


def test_workflow_notary_language_validation_branch():
    invalid_payload = SimpleNamespace(text="hello", language="fr")
    with pytest.raises(Exception):
        wf_router._validate_notary_request(invalid_payload, "tenant-1")


@pytest.mark.asyncio
async def test_ask_handler_success_timeout_and_fallback(monkeypatch):
    handler = wf_router._make_ask_handler(_tenant_getter)
    payload = AskRequest(question="q", context="ctx")

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )

    service = SimpleNamespace(
        ask_flow=AsyncMock(
            return_value={"answer": "A", "model": "m", "source": "llm", "metadata": {"x": 1}}
        )
    )
    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: service)
    out = await handler(payload, "tenant-1", object())
    assert out.answer == "A"

    original_wait_for = wf_router.asyncio.wait_for

    async def _raise_timeout(awaitable, **_kwargs):
        await asyncio.sleep(0)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", _raise_timeout)
    with pytest.raises(HTTPException) as timeout_exc:
        await handler(payload, "tenant-1", object())
    assert timeout_exc.value.status_code == 504

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", original_wait_for)

    failing_service = SimpleNamespace(ask_flow=AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service", lambda db: failing_service
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_ask_flow",
        AsyncMock(
            return_value={"answer": "fallback", "model": "f", "source": "fallback", "metadata": {}}
        ),
    )
    fallback = await handler(payload, "tenant-1", object())
    assert fallback["answer"] == "fallback"


@pytest.mark.asyncio
async def test_classify_handler_success_timeout_and_fallback(monkeypatch):
    handler = wf_router._make_classify_handler(_tenant_getter)
    payload = ClassifyRequest(text="hello", candidate_labels=["invoice", "other"])

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )

    service = SimpleNamespace(
        classify_flow=AsyncMock(
            return_value={
                "predicted_category": "invoice",
                "confidence_score": 0.9,
                "model": "m",
                "source": "llm",
                "metadata": {},
            }
        )
    )
    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: service)
    out = await handler(payload, "tenant-1", object())
    assert out.label == "invoice"

    original_wait_for = wf_router.asyncio.wait_for

    async def _raise_timeout(awaitable, **_kwargs):
        await asyncio.sleep(0)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", _raise_timeout)
    with pytest.raises(HTTPException) as timeout_exc:
        await handler(payload, "tenant-1", object())
    assert timeout_exc.value.status_code == 504

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", original_wait_for)

    failing_service = SimpleNamespace(classify_flow=AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service", lambda db: failing_service
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_classify_flow",
        AsyncMock(
            return_value={
                "label": "other",
                "confidence": 0.5,
                "model": "f",
                "source": "fallback",
                "metadata": {},
            }
        ),
    )
    fallback = await handler(payload, "tenant-1", object())
    assert fallback["label"] == "other"


@pytest.mark.asyncio
async def test_notary_handler_success_timeout_and_fallback(monkeypatch):
    handler = wf_router._make_notary_handler(_tenant_getter)
    payload = NotarySummarizeRequest(text="hello", language="nl")

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )

    service = SimpleNamespace(
        summarize_flow=AsyncMock(
            return_value={"summary": "S", "key_points": ["k1"], "source": "llm", "metadata": {}}
        )
    )
    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: service)
    out = await handler(payload, "tenant-1", object())
    assert out.summary.raw_summary == "S"

    original_wait_for = wf_router.asyncio.wait_for

    async def _raise_timeout(awaitable, **_kwargs):
        await asyncio.sleep(0)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", _raise_timeout)
    with pytest.raises(HTTPException) as timeout_exc:
        await handler(payload, "tenant-1", object())
    assert timeout_exc.value.status_code == 504

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", original_wait_for)

    failing_service = SimpleNamespace(summarize_flow=AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service", lambda db: failing_service
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_notary_summarization_flow",
        AsyncMock(
            return_value={
                "document_id": None,
                "summary": {
                    "title": "Summary",
                    "key_points": ["k"],
                    "parties_involved": [],
                    "risks_or_warnings": [],
                    "raw_summary": "fallback",
                },
                "source": "fallback",
                "metadata": {},
            }
        ),
    )
    fallback = await handler(payload, "tenant-1", object())
    assert fallback["source"] == "fallback"


@pytest.mark.asyncio
async def test_notary_handler_document_lookup_and_error_mapping(monkeypatch):
    handler = wf_router._make_notary_handler(_tenant_getter)
    payload = NotarySummarizeRequest(document_id="doc-1", text="hello", language="nl")

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )
    monkeypatch.setattr(
        "app.http.routers.workflows._validate_notary_request", lambda payload, tenant_id: None
    )

    class _DocRepo:
        def __init__(self, db):
            self.db = db

        async def read(self, doc_id, tenant_id):
            await asyncio.sleep(0)
            return SimpleNamespace(text="from-doc")

    monkeypatch.setattr("app.http.routers.workflows.DocumentRepository", _DocRepo)
    service = SimpleNamespace(
        summarize_flow=AsyncMock(
            return_value={"summary": "S", "key_points": [], "source": "llm", "metadata": {}}
        )
    )
    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: service)

    out = await handler(payload, "tenant-1", object())
    assert out.summary.raw_summary == "S"
    service.summarize_flow.assert_awaited_once()
    assert service.summarize_flow.await_args.args[0] == "from-doc"

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_notary_request",
        lambda payload, tenant_id: _raise_app_error("bad", 422),
    )
    with pytest.raises(HTTPException) as app_exc:
        await handler(payload, "tenant-1", object())
    assert app_exc.value.status_code == 422

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_notary_request", lambda payload, tenant_id: None
    )
    failing_service = SimpleNamespace(summarize_flow=AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service", lambda db: failing_service
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_notary_summarization_flow",
        AsyncMock(side_effect=RuntimeError("fallback failed")),
    )
    with pytest.raises(HTTPException) as generic_exc:
        await handler(payload, "tenant-1", object())
    assert generic_exc.value.status_code == 500


@pytest.mark.asyncio
async def test_notary_handler_keeps_original_text_when_document_missing(monkeypatch):
    handler = wf_router._make_notary_handler(_tenant_getter)
    payload = NotarySummarizeRequest(document_id="doc-1", text="original", language="nl")

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )
    monkeypatch.setattr(
        "app.http.routers.workflows._validate_notary_request", lambda payload, tenant_id: None
    )

    class _DocRepo:
        def __init__(self, db):
            self.db = db

        async def read(self, doc_id, tenant_id):
            await asyncio.sleep(0)
            return None

    monkeypatch.setattr("app.http.routers.workflows.DocumentRepository", _DocRepo)
    service = SimpleNamespace(
        summarize_flow=AsyncMock(
            return_value={"summary": "S", "key_points": [], "source": "llm", "metadata": {}}
        )
    )
    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: service)

    await handler(payload, "tenant-1", object())
    assert service.summarize_flow.await_args.args[0] == "original"


def test_ask_stream_handler_config_and_validation_errors(monkeypatch):
    handler = wf_router._make_ask_stream_handler(_tenant_getter)
    payload = AskRequest(question="q", context="ctx")

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: False
    )
    with pytest.raises(HTTPException) as config_exc:
        handler(payload, "tenant-1", object())
    assert config_exc.value.status_code == 400

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )
    bad_payload = AskRequest(question=" ", context="ctx")
    with pytest.raises(HTTPException) as validation_exc:
        handler(bad_payload, "tenant-1", object())
    assert validation_exc.value.status_code == 400


@pytest.mark.asyncio
async def test_handlers_llm_not_configured_return_400(monkeypatch):
    ask_handler = wf_router._make_ask_handler(_tenant_getter)
    classify_handler = wf_router._make_classify_handler(_tenant_getter)
    notary_handler = wf_router._make_notary_handler(_tenant_getter)

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: False
    )

    with pytest.raises(HTTPException) as ask_exc:
        await ask_handler(AskRequest(question="q", context="ctx"), "tenant-1", object())
    assert ask_exc.value.status_code == 400

    with pytest.raises(HTTPException) as classify_exc:
        await classify_handler(
            ClassifyRequest(text="t", candidate_labels=["a"]),
            "tenant-1",
            object(),
        )
    assert classify_exc.value.status_code == 400

    with pytest.raises(HTTPException) as notary_exc:
        await notary_handler(NotarySummarizeRequest(text="t", language="nl"), "tenant-1", object())
    assert notary_exc.value.status_code == 400


@pytest.mark.asyncio
async def test_handlers_validation_errors_are_mapped_to_http_400(monkeypatch):
    ask_handler = wf_router._make_ask_handler(_tenant_getter)
    classify_handler = wf_router._make_classify_handler(_tenant_getter)
    notary_handler = wf_router._make_notary_handler(_tenant_getter)

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )

    with pytest.raises(HTTPException) as ask_exc:
        await ask_handler(AskRequest(question=" ", context="ctx"), "tenant-1", object())
    assert ask_exc.value.status_code == 400

    with pytest.raises(HTTPException) as classify_exc:
        await classify_handler(
            ClassifyRequest(text=" ", candidate_labels=["a"]), "tenant-1", object()
        )
    assert classify_exc.value.status_code == 400

    with pytest.raises(HTTPException) as notary_exc:
        await notary_handler(NotarySummarizeRequest(text=" ", language="nl"), "tenant-1", object())
    assert notary_exc.value.status_code == 400


@pytest.mark.asyncio
async def test_handlers_map_app_error_and_generic_error(monkeypatch):
    ask_handler = wf_router._make_ask_handler(_tenant_getter)
    classify_handler = wf_router._make_classify_handler(_tenant_getter)

    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_ask_request",
        lambda payload, tenant_id: _raise_app_error("denied", 418),
    )
    with pytest.raises(HTTPException) as app_exc:
        await ask_handler(AskRequest(question="q", context="ctx"), "tenant-1", object())
    assert app_exc.value.status_code == 418

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_ask_request", lambda payload, tenant_id: None
    )
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service",
        lambda db: SimpleNamespace(ask_flow=AsyncMock(side_effect=RuntimeError("boom"))),
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_ask_flow",
        AsyncMock(side_effect=RuntimeError("fallback failed")),
    )
    with pytest.raises(HTTPException) as generic_ask_exc:
        await ask_handler(AskRequest(question="q", context="ctx"), "tenant-1", object())
    assert generic_ask_exc.value.status_code == 500

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_classify_request",
        lambda payload, tenant_id: _raise_app_error("bad", 409),
    )
    with pytest.raises(HTTPException) as classify_app_exc:
        await classify_handler(
            ClassifyRequest(text="t", candidate_labels=["a"]), "tenant-1", object()
        )
    assert classify_app_exc.value.status_code == 409

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_classify_request", lambda payload, tenant_id: None
    )
    monkeypatch.setattr(
        "app.http.routers.workflows._build_workflow_service",
        lambda db: SimpleNamespace(classify_flow=AsyncMock(side_effect=RuntimeError("boom"))),
    )
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.run_classify_flow",
        AsyncMock(side_effect=RuntimeError("fallback failed")),
    )
    with pytest.raises(HTTPException) as generic_classify_exc:
        await classify_handler(
            ClassifyRequest(text="t", candidate_labels=["a"]), "tenant-1", object()
        )
    assert generic_classify_exc.value.status_code == 500


@pytest.mark.asyncio
async def test_ask_stream_handler_timeout_and_error_mapping(monkeypatch):
    handler = wf_router._make_ask_stream_handler(_tenant_getter)
    payload = AskRequest(question="q", context="ctx")
    monkeypatch.setattr(
        "app.http.routers.workflows.services_ai_flows.llm_client.is_configured", lambda: True
    )
    monkeypatch.setattr(
        "app.http.routers.workflows._validate_ask_request", lambda payload, tenant_id: None
    )

    async def _never_finishes(*args, **kwargs):
        yield "token"

    monkeypatch.setattr("app.http.routers.workflows.run_ask_flow_stream", _never_finishes)

    original_wait_for = wf_router.asyncio.wait_for

    async def _timeout(awaitable, **_kwargs):
        await asyncio.sleep(0)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", _timeout)
    response = handler(payload, "tenant-1", object())
    chunks = [chunk async for chunk in response.body_iterator]
    joined = b"".join(c if isinstance(c, bytes) else str(c).encode() for c in chunks).decode()
    assert "Stream operation timed out" in joined
    monkeypatch.setattr("app.http.routers.workflows.asyncio.wait_for", original_wait_for)

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_ask_request",
        lambda payload, tenant_id: _raise_app_error("bad", 422),
    )
    with pytest.raises(HTTPException) as app_exc:
        handler(payload, "tenant-1", object())
    assert app_exc.value.status_code == 422

    monkeypatch.setattr(
        "app.http.routers.workflows._validate_ask_request", lambda payload, tenant_id: None
    )
    monkeypatch.setattr("app.http.routers.workflows.stream_text_tokens", _raise_runtime_error)
    with pytest.raises(HTTPException) as generic_exc:
        handler(payload, "tenant-1", object())
    assert generic_exc.value.status_code == 500


@pytest.mark.asyncio
async def test_run_ask_flow_stream_uses_service(monkeypatch):
    class _Service:
        async def ask_flow_stream(self, question, document_ids, user_context, context):
            yield "A"
            yield "B"

    monkeypatch.setattr("app.http.routers.workflows._build_workflow_service", lambda db: _Service())
    payload = AskRequest(question="q", context="ctx")

    tokens = []
    async for t in wf_router.run_ask_flow_stream(
        tenant_id="tenant-1", db=object(), payload=payload
    ):
        tokens.append(t)
    assert "".join(tokens) == "AB"


def test_rag_ensure_llm_configured_branch(monkeypatch):
    monkeypatch.setattr(
        "app.http.routers.rag.get_settings",
        lambda: SimpleNamespace(llm_provider="", llm_base_url=None),
    )
    with pytest.raises(HTTPException) as exc:
        rag_router._ensure_llm_configured()
    assert exc.value.status_code == 400


def test_build_rag_service_uses_registry(monkeypatch):
    fake_settings = SimpleNamespace()
    monkeypatch.setattr("app.http.routers.rag.get_settings", lambda: fake_settings)

    class _Registry:
        def __init__(self, settings):
            self.settings = settings

        def get_llm_provider(self):
            return "llm-provider"

        def get_embedding_provider(self):
            return "embedding-provider"

        def get_search_provider(self):
            return "search-provider"

    monkeypatch.setattr("app.http.routers.rag.ProviderRegistry", _Registry)
    monkeypatch.setattr(
        "app.http.routers.rag.LLMService", lambda provider: ("llm-service", provider)
    )
    monkeypatch.setattr(
        "app.http.routers.rag.RAGService",
        lambda llm_service, embedding_provider, search_provider, document_repository: (
            llm_service,
            embedding_provider,
            search_provider,
            type(document_repository).__name__,
        ),
    )

    rag_service, llm_service = rag_router._build_rag_service(object())
    assert llm_service == ("llm-service", "llm-provider")
    assert rag_service[1] == "embedding-provider"
    assert rag_service[2] == "search-provider"


@pytest.mark.asyncio
async def test_rag_index_route_direct_invocation(monkeypatch):
    router = rag_router.build_rag_router(_tenant_getter)
    rag_index = next(route.endpoint for route in router.routes if route.path == "/ai/rag/index")

    payload = SimpleNamespace(document_id="doc-1")

    monkeypatch.setattr("app.http.routers.rag.fetch_document", AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as not_found_exc:
        await rag_index(payload, "tenant-1", object())
    assert not_found_exc.value.status_code == 404

    monkeypatch.setattr(
        "app.http.routers.rag.fetch_document",
        AsyncMock(return_value=SimpleNamespace(text="doc text")),
    )
    monkeypatch.setattr(
        "app.http.routers.rag.rag_pipeline.index_document", AsyncMock(return_value=3)
    )
    out = await rag_index(payload, "tenant-1", object())
    assert out.document_id == "doc-1"
    assert out.chunks_indexed == 3


@pytest.mark.asyncio
async def test_run_rag_query_flow_and_stream_branches(monkeypatch):
    docs = [{"document_id": "d1", "text": "ctx", "score": 0.9}]
    rag_service = SimpleNamespace(
        retrieve_documents=AsyncMock(return_value=docs),
        answer_question=AsyncMock(return_value={"answer": "A", "model": "m", "latency_ms": 3}),
    )

    async def _stream_complete(prompt, system_prompt, context):
        yield "X"
        yield "Y"

    llm_service = SimpleNamespace(stream_complete=_stream_complete)
    monkeypatch.setattr(
        "app.http.routers.rag._build_rag_service", lambda db: (rag_service, llm_service)
    )

    payload = RAGQueryRequest(query="What?", top_k=2)
    out = await rag_router.run_rag_query_flow(tenant_id="tenant-1", db=object(), payload=payload)
    assert out["answer"] == "A"
    assert out["sources"][0]["document_id"] == "d1"

    tokens = []
    async for t in rag_router.run_rag_query_flow_stream(
        tenant_id="tenant-1",
        db=object(),
        payload=payload,
    ):
        tokens.append(t)
    assert "".join(tokens) == "XY"

    empty_rag_service = SimpleNamespace(retrieve_documents=AsyncMock(return_value=[]))
    captured = {}

    async def _capturing_stream(prompt, system_prompt, context):
        captured["prompt"] = prompt
        yield "Z"

    monkeypatch.setattr(
        "app.http.routers.rag._build_rag_service",
        lambda db: (empty_rag_service, SimpleNamespace(stream_complete=_capturing_stream)),
    )

    empty_tokens = []
    async for t in rag_router.run_rag_query_flow_stream(
        tenant_id="tenant-1",
        db=object(),
        payload=payload,
    ):
        empty_tokens.append(t)

    assert "".join(empty_tokens) == "Z"
    assert "(No relevant documents found.)" in captured["prompt"]
