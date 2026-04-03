from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy.exc import IntegrityError

from app.documents.service import (
    DocumentConflictError,
    UploadTooLargeError,
    UploadValidationError,
    _is_duplicate_key_error,
    _message_indicates_duplicate,
    create_document,
    document_to_read,
    fetch_document,
    fetch_document_payload,
    prepare_uploaded_document,
)
from app.llm.errors import LLMError, LLMProviderError, LLMTimeoutError
from app.llm.providers import OpenAICompatibleProvider, OllamaProvider
from app.models import AiCallAudit, Document
from app.persistence.repositories.audit import AuditRepository
from app.persistence.repositories.document import DocumentRepository
from app.providers.search import SearchError
from app.providers.search_impl.duckduckgo import DuckDuckGoSearchProvider
from app.providers.search_impl.mock import MockSearchProvider
from app.providers.search_impl.tavily import TavilySearchProvider


def test_main_and_orchestration_placeholders_importable():
    import app.main as main_module
    from app.orchestration import agent_runtime, planner, workflow_engine

    assert main_module.__all__ == ["app"]
    assert agent_runtime.__all__ == []
    assert planner.__all__ == []
    assert workflow_engine.__all__ == []


class _FakeUpload:
    def __init__(self, filename: str, body: bytes):
        self.filename = filename
        self._body = body

    async def read(self) -> bytes:
        await asyncio.sleep(0)
        return self._body


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


def _settings(**overrides: Any):
    from app.core.config import Settings

    base = {
        "llm_provider": "ollama",
        "llm_base_url": "http://localhost:11434",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 1,
        "llm_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_document_helpers_create_fetch_payload_and_conflict(db_session):
    created = await create_document(
        db_session,
        "tenant-1",
        document_id="doc-1",
        title="Title",
        text="Body",
    )
    assert created.id == "doc-1"

    read_back = await fetch_document(db_session, "tenant-1", "doc-1")
    payload = await fetch_document_payload(db_session, "tenant-1", "doc-1")
    assert read_back is not None
    assert payload == {"id": "doc-1", "title": "Title", "text": "Body"}

    with pytest.raises(DocumentConflictError):
        await create_document(
            db_session,
            "tenant-1",
            document_id="doc-1",
            title="Again",
            text="Again",
        )


@pytest.mark.asyncio
async def test_document_to_read_and_upload_validation_paths():
    doc = SimpleNamespace(id="d1", title="T", text="X", created_at=None)
    out = document_to_read(doc)
    assert out.created_at.tzinfo is not None

    prepared = await prepare_uploaded_document(
        _FakeUpload("demo.txt", b"hello"),
        document_id="doc-x",
        title="My title",
    )
    assert prepared.document_id == "doc-x"
    assert prepared.text == "hello"

    with pytest.raises(UploadValidationError):
        await prepare_uploaded_document(_FakeUpload("bad.txt", b"\xff\xfe"))

    with pytest.raises(UploadTooLargeError):
        await prepare_uploaded_document(_FakeUpload("big.txt", b"a" * (6 * 1024 * 1024)))

    with pytest.raises(UploadValidationError):
        await prepare_uploaded_document(_FakeUpload("x" * 70 + ".txt", b"ok"))


def test_duplicate_message_helpers_cover_branches():
    assert _message_indicates_duplicate("duplicate key value violates unique constraint") is True
    assert _message_indicates_duplicate("NOT NULL constraint failed") is False
    assert _is_duplicate_key_error(RuntimeError("duplicate entry")) is True
    assert _is_duplicate_key_error(RuntimeError("foreign key violation")) is False

    err_sqlstate = IntegrityError("stmt", {}, SimpleNamespace(sqlstate="23505"))
    assert _is_duplicate_key_error(err_sqlstate) is True

    err_pgcode = IntegrityError("stmt", {}, SimpleNamespace(pgcode="23505"))
    assert _is_duplicate_key_error(err_pgcode) is True

    unique_orig = type("UniqueViolationError", (), {})()
    err_unique = IntegrityError("stmt", {}, unique_orig)
    assert _is_duplicate_key_error(err_unique) is True

    cause = type("UniqueViolationError", (), {})()
    orig_with_cause = SimpleNamespace(__cause__=cause)
    err_cause = IntegrityError("stmt", {}, orig_with_cause)
    assert _is_duplicate_key_error(err_cause) is True


@pytest.mark.asyncio
async def test_document_repository_crud_and_filters(db_session):
    repo = DocumentRepository(db_session)
    d1 = Document(id="d1", tenant_id="tenant-1", title="Alpha", text="A")
    d2 = Document(id="d2", tenant_id="tenant-1", title="Beta", text="B")
    await repo.create(d1)
    await repo.create(d2)

    got = await repo.read("d1", tenant_id="tenant-1")
    assert got is not None
    assert got.title == "Alpha"

    got.title = "Alpha Updated"
    updated = await repo.update(got)
    assert updated.title == "Alpha Updated"

    filtered = await repo.list("tenant-1", title_contains="Beta")
    assert len(filtered) == 1
    assert filtered[0].id == "d2"

    deleted = await repo.delete("d2", tenant_id="tenant-1")
    assert deleted is True
    assert await repo.read("d2", tenant_id="tenant-1") is None
    assert await repo.read("d1", tenant_id=None) is None
    assert await repo.delete("d1", tenant_id=None) is False


@pytest.mark.asyncio
async def test_audit_repository_crud_list_and_purge(db_session):
    repo = AuditRepository(db_session)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    old = now - timedelta(days=40)

    a1 = AiCallAudit(
        id="a1",
        tenant_id="tenant-1",
        flow_name="ask",
        request_payload={"q": "1"},
        response_payload={"a": "1"},
        success=True,
        created_at=old,
    )
    a2 = AiCallAudit(
        id="a2",
        tenant_id="tenant-1",
        flow_name="ask",
        request_payload={"q": "2"},
        response_payload={"a": "2"},
        success=False,
        created_at=now,
    )
    await repo.create(a1)
    await repo.create(a2)

    got = await repo.read("a1", tenant_id="tenant-1")
    assert got is not None
    assert got.id == "a1"
    assert await repo.read("a1", tenant_id=None) is None

    got.response_payload = {"a": "updated"}
    updated = await repo.update(got)
    assert updated.response_payload == {"a": "updated"}

    all_for_tenant = await repo.list("tenant-1")
    assert len(all_for_tenant) == 2

    ask_only = await repo.list("tenant-1", flow_name="ask")
    assert len(ask_only) == 2

    success_only = await repo.list("tenant-1", success_only=True)
    assert len(success_only) == 1
    assert success_only[0].id == "a1"

    created_after = await repo.list("tenant-1", created_after=now - timedelta(days=1))
    assert len(created_after) == 1
    assert created_after[0].id == "a2"

    deleted_one = await repo.delete("a2", tenant_id="tenant-1")
    assert deleted_one is True
    assert await repo.delete("a2", tenant_id=None) is False

    purged_tenant = await repo.purge_older_than("tenant-1", now - timedelta(days=30))
    assert purged_tenant == 1


@pytest.mark.asyncio
async def test_mock_search_provider_paths():
    provider = MockSearchProvider()

    python_results = await provider.search("python tutorial", limit=5)
    assert len(python_results) >= 1

    generic_results = await provider.search("unmatched query", limit=1)
    assert len(generic_results) == 1
    assert provider.get_name() == "mock"


@pytest.mark.asyncio
async def test_duckduckgo_search_success_and_failures(monkeypatch):
    provider = DuckDuckGoSearchProvider()

    async def _good_fetch(query: str):
        await asyncio.sleep(0)
        return {
            "RelatedTopics": [
                {"Text": "Doc - snippet", "FirstURL": "https://example.com/doc"},
            ]
        }

    monkeypatch.setattr("app.providers.search_impl.duckduckgo._fetch_ddg_payload", _good_fetch)
    ok = await provider.search("doc", limit=5)
    assert len(ok) == 1

    async def _request_error(query: str):
        raise httpx.RequestError("network")

    monkeypatch.setattr("app.providers.search_impl.duckduckgo._fetch_ddg_payload", _request_error)
    with pytest.raises(SearchError):
        await provider.search("doc", limit=5)

    async def _bad_json(query: str):
        raise ValueError("bad")

    monkeypatch.setattr("app.providers.search_impl.duckduckgo._fetch_ddg_payload", _bad_json)
    with pytest.raises(SearchError):
        await provider.search("doc", limit=5)


@pytest.mark.asyncio
async def test_tavily_search_success_and_failures(monkeypatch):
    provider = TavilySearchProvider(api_key="k")

    class _FakeClient:
        def __init__(self, response=None, error: Exception | None = None):
            self._response = response
            self._error = error

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict[str, Any]):
            await asyncio.sleep(0)
            if self._error:
                raise self._error
            return self._response

    monkeypatch.setattr(
        "app.providers.search_impl.tavily.httpx.AsyncClient",
        lambda timeout=10.0: _FakeClient(
            _FakeResponse(
                200,
                payload={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://x",
                            "content": "S",
                            "score": 0.9,
                        }
                    ]
                },
            )
        ),
    )
    ok = await provider.search("q", limit=5)
    assert len(ok) == 1
    assert ok[0].metadata["source"] == "tavily"

    monkeypatch.setattr(
        "app.providers.search_impl.tavily.httpx.AsyncClient",
        lambda timeout=10.0: _FakeClient(_FakeResponse(500, text="oops")),
    )
    with pytest.raises(SearchError):
        await provider.search("q", limit=5)

    monkeypatch.setattr(
        "app.providers.search_impl.tavily.httpx.AsyncClient",
        lambda timeout=10.0: _FakeClient(error=httpx.RequestError("down")),
    )
    with pytest.raises(SearchError):
        await provider.search("q", limit=5)

    class _BadJsonResponse(_FakeResponse):
        def json(self):
            raise ValueError("bad")

    monkeypatch.setattr(
        "app.providers.search_impl.tavily.httpx.AsyncClient",
        lambda timeout=10.0: _FakeClient(_BadJsonResponse(200)),
    )
    with pytest.raises(SearchError):
        await provider.search("q", limit=5)


@pytest.mark.asyncio
async def test_legacy_llm_complete_paths(monkeypatch):
    ollama = OllamaProvider(_settings())
    openai = OpenAICompatibleProvider(_settings(llm_provider="openai"))

    async def _ok_ollama(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"response": "hello", "model": "m"})

    monkeypatch.setattr("app.llm.providers._post_with_retries", _ok_ollama)
    ollama_out = await ollama.complete("prompt")
    assert ollama_out.raw_text == "hello"

    async def _timeout(**kwargs):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr("app.llm.providers._post_with_retries", _timeout)
    with pytest.raises(LLMTimeoutError):
        await ollama.complete("prompt")

    async def _request_err(**kwargs):
        raise httpx.RequestError("down")

    monkeypatch.setattr("app.llm.providers._post_with_retries", _request_err)
    with pytest.raises(LLMProviderError):
        await ollama.complete("prompt")

    async def _bad_status_ollama(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(500, text="oops")

    monkeypatch.setattr("app.llm.providers._post_with_retries", _bad_status_ollama)
    with pytest.raises(LLMProviderError):
        await ollama.complete("prompt")

    async def _empty_ollama(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"response": " "})

    monkeypatch.setattr("app.llm.providers._post_with_retries", _empty_ollama)
    with pytest.raises(LLMProviderError):
        await ollama.complete("prompt")

    async def _ok_openai(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(
            200,
            payload={"model": "gpt", "choices": [{"message": {"content": "hi"}}]},
        )

    monkeypatch.setattr("app.llm.providers._post_with_retries", _ok_openai)
    openai_out = await openai.complete("prompt")
    assert openai_out.raw_text == "hi"

    async def _bad_openai_status(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(429, text="rate")

    monkeypatch.setattr("app.llm.providers._post_with_retries", _bad_openai_status)
    with pytest.raises(LLMProviderError):
        await openai.complete("prompt")

    async def _no_choices(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"choices": []})

    monkeypatch.setattr("app.llm.providers._post_with_retries", _no_choices)
    with pytest.raises(LLMProviderError):
        await openai.complete("prompt")

    async def _empty_content(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"choices": [{"message": {"content": " "}}]})

    monkeypatch.setattr("app.llm.providers._post_with_retries", _empty_content)
    with pytest.raises(LLMProviderError):
        await openai.complete("prompt")


@pytest.mark.asyncio
async def test_legacy_llm_stream_error_paths(monkeypatch):
    ollama = OllamaProvider(_settings())
    openai = OpenAICompatibleProvider(_settings(llm_provider="openai"))

    class _StreamResponse:
        def __init__(self, status_code: int, lines: list[str] | None = None, body: bytes = b"err"):
            self.status_code = status_code
            self._lines = lines or []
            self._body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self) -> bytes:
            await asyncio.sleep(0)
            return self._body

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _Client:
        def __init__(self, response=None, error: Exception | None = None):
            self._response = response
            self._error = error

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            if self._error:
                raise self._error
            return self._response

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(
            _StreamResponse(200, lines=['{"response":"A"}', "invalid-json", '{"text":"B"}'])
        ),
    )
    chunks = [chunk async for chunk in ollama.stream_complete("p")]
    assert "".join(chunks) == "AB"

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(_StreamResponse(500)),
    )
    with pytest.raises(LLMError):
        [chunk async for chunk in ollama.stream_complete("p")]

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(error=httpx.RequestError("down")),
    )
    with pytest.raises(LLMError):
        [chunk async for chunk in ollama.stream_complete("p")]

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(
            _StreamResponse(
                200,
                lines=[
                    'data: {"choices":[{"delta":{"content":"A"}}]}',
                    "data: [DONE]",
                ],
            )
        ),
    )
    openai_chunks = [chunk async for chunk in openai.stream_complete("p")]
    assert "".join(openai_chunks) == "A"

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(_StreamResponse(500)),
    )
    with pytest.raises(LLMError):
        [chunk async for chunk in openai.stream_complete("p")]

    monkeypatch.setattr(
        "app.llm.providers.httpx.AsyncClient",
        lambda timeout=1: _Client(error=httpx.RequestError("down")),
    )
    with pytest.raises(LLMError):
        [chunk async for chunk in openai.stream_complete("p")]
