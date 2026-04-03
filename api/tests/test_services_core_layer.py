from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.context import ExecutionContext
from app.core.errors import NotFoundError, ValidationError
from app.domain.memory import MemoryType
from app.domain.tool import ToolOutput, ToolType
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.tool_service import ToolService
from app.services.workflow_service import WorkflowService


@pytest.mark.asyncio
async def test_memory_service_store_and_retrieve_roundtrip():
    service = MemoryService(ttl_seconds=60)
    ctx = ExecutionContext.from_request("tenant-1")

    await service.store("agent-1", MemoryType.SHORT_TERM, "k", "v", context=ctx)
    value = await service.retrieve("agent-1", MemoryType.SHORT_TERM, "k", context=ctx)

    assert value == "v"


@pytest.mark.asyncio
async def test_memory_service_retrieve_missing_returns_none():
    service = MemoryService()
    ctx = ExecutionContext.from_request("tenant-1")

    value = await service.retrieve("missing-agent", MemoryType.SHORT_TERM, "missing", context=ctx)

    assert value is None


@pytest.mark.asyncio
async def test_memory_service_clear_only_target_type():
    service = MemoryService()
    ctx = ExecutionContext.from_request("tenant-1")

    await service.store("agent-1", MemoryType.SHORT_TERM, "short", "s", context=ctx)
    await service.store("agent-1", MemoryType.LONG_TERM, "long", "l", context=ctx)

    await service.clear("agent-1", MemoryType.SHORT_TERM, context=ctx)

    short_val = await service.retrieve("agent-1", MemoryType.SHORT_TERM, "short", context=ctx)
    long_val = await service.retrieve("agent-1", MemoryType.LONG_TERM, "long", context=ctx)
    assert short_val is None
    assert long_val == "l"


@pytest.mark.asyncio
async def test_memory_service_cleanup_expired_removes_entries():
    service = MemoryService()
    ctx = ExecutionContext.from_request("tenant-1")

    await service.store("agent-1", MemoryType.SHORT_TERM, "alive", "v", context=ctx)
    await service.store("agent-1", MemoryType.SHORT_TERM, "expired", "x", context=ctx)

    memory = service._storage[str(ctx.tenant_id)]["agent-1"]
    memory.entries["expired"].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    removed = await service.cleanup_expired("agent-1", MemoryType.SHORT_TERM, context=ctx)

    assert removed == 1
    assert "expired" not in memory.entries
    assert "alive" in memory.entries


@pytest.mark.asyncio
async def test_memory_service_summary_counts_by_type():
    service = MemoryService()
    ctx = ExecutionContext.from_request("tenant-1")

    await service.store("agent-1", MemoryType.SHORT_TERM, "k1", "v1", context=ctx)
    await service.store("agent-1", MemoryType.PREFERENCE, "k2", "v2", context=ctx)

    summary = await service.get_agent_summary("agent-1", context=ctx)

    assert summary["agent_id"] == "agent-1"
    assert summary["memory_types"]["short_term"] == 1
    assert summary["memory_types"]["preference"] == 1


@pytest.mark.asyncio
async def test_memory_service_requires_context_when_not_provided():
    service = MemoryService()

    with pytest.raises(RuntimeError, match="Execution context is not set"):
        await service.store("agent-1", MemoryType.SHORT_TERM, "k", "v")


def _make_tool_service() -> ToolService:
    return ToolService()


def _ctx() -> ExecutionContext:
    return ExecutionContext.from_request("tenant-1")


def test_tool_service_register_and_get_tool():
    service = _make_tool_service()

    def add(a: int, b: int) -> int:
        return a + b

    service.register_tool("t1", "adder", "add values", {"type": "object"}, add)
    tool = service.get_tool("t1")

    assert tool.definition.name == "adder"


def test_tool_service_register_duplicate_raises_validation_error():
    service = _make_tool_service()

    def fn() -> str:
        return "ok"

    service.register_tool("dup", "tool", "desc", {}, fn)
    with pytest.raises(ValidationError):
        service.register_tool("dup", "tool", "desc", {}, fn)


def test_tool_service_get_missing_raises_not_found():
    service = _make_tool_service()

    with pytest.raises(NotFoundError):
        service.get_tool("missing")


def test_tool_service_list_tools_with_filter():
    service = _make_tool_service()

    def fn() -> str:
        return "ok"

    service.register_tool("f", "func", "desc", {}, fn, tool_type=ToolType.FUNCTION)
    service.register_tool("s", "search", "desc", {}, fn, tool_type=ToolType.SEARCH)

    function_tools = service.list_tools(tool_type=ToolType.FUNCTION)
    all_tools = service.list_tools()

    assert len(function_tools) == 1
    assert len(all_tools) == 2


@pytest.mark.asyncio
async def test_tool_service_execute_tool_returns_tool_output():
    service = _make_tool_service()

    def add(a: int, b: int) -> int:
        return a + b

    service.register_tool("t1", "adder", "add values", {}, add)

    result = await service.execute_tool("t1", {"a": 2, "b": 3}, context=_ctx())

    assert isinstance(result, ToolOutput)
    assert result.success is True
    assert result.value == 5


@pytest.mark.asyncio
async def test_tool_service_execute_tool_with_async_callable():
    service = _make_tool_service()

    async def add_async(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    service.register_tool("t1", "adder", "add values", {}, add_async)

    result = await service.execute_tool("t1", {"a": 4, "b": 6}, context=_ctx())

    assert result.success is True
    assert result.value == 10


@pytest.mark.asyncio
async def test_tool_service_execute_tool_requires_context_when_not_provided():
    service = _make_tool_service()

    def fn() -> str:
        return "ok"

    service.register_tool("t1", "tool", "desc", {}, fn)

    with pytest.raises(RuntimeError, match="Execution context is not set"):
        await service.execute_tool("t1", {})


def test_tool_service_get_tools_for_agent_skips_missing():
    service = _make_tool_service()

    def fn() -> str:
        return "ok"

    service.register_tool("present", "tool", "desc", {}, fn)

    tools = service.get_tools_for_agent(["present", "missing"])

    assert len(tools) == 1
    assert tools[0].definition.id == "present"


def test_tool_service_unregister_tool():
    service = _make_tool_service()

    def fn() -> str:
        return "ok"

    service.register_tool("t1", "tool", "desc", {}, fn)
    service.unregister_tool("t1")

    with pytest.raises(NotFoundError):
        service.get_tool("t1")


def test_tool_service_unregister_missing_raises_not_found():
    service = _make_tool_service()

    with pytest.raises(NotFoundError):
        service.unregister_tool("missing")


class _FakeRagService:
    def __init__(self):
        self.retrieve_documents = AsyncMock()
        self.answer_question = AsyncMock()
        self.stream_answer = None
        self.complete_text = AsyncMock()


class _FakeAuditService:
    def __init__(self):
        self.record_flow_execution = AsyncMock()


@pytest.mark.asyncio
async def test_workflow_ask_flow_uses_user_context_in_retrieval_query():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.retrieve_documents.return_value = [{"document_id": "d1", "title": "T", "text": "X"}]
    rag.answer_question.return_value = {"answer": "A", "model": "m", "latency_ms": 1, "sources": []}

    service = WorkflowService(rag, audit)
    ctx = _ctx()

    out = await service.ask_flow("question", user_context="context", context=ctx)

    assert out["answer"] == "A"
    retrieval_query = rag.retrieve_documents.await_args.args[0]
    assert "question" in retrieval_query
    assert "context" in retrieval_query


@pytest.mark.asyncio
async def test_workflow_ask_flow_records_failure_audit_on_error():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.retrieve_documents.side_effect = RuntimeError("boom")

    service = WorkflowService(rag, audit)

    with pytest.raises(RuntimeError, match="boom"):
        await service.ask_flow("q", context=_ctx())

    assert audit.record_flow_execution.await_count == 1
    kwargs = audit.record_flow_execution.await_args.kwargs
    assert kwargs["success"] is False


@pytest.mark.asyncio
async def test_workflow_ask_flow_stream_yields_tokens():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.retrieve_documents.return_value = [{"document_id": "d1", "title": "T", "text": "X"}]

    async def _stream(*args, **kwargs):
        yield "A"
        yield "B"

    rag.stream_answer = _stream
    service = WorkflowService(rag, audit)

    tokens = []
    async for token in service.ask_flow_stream("q", context=_ctx()):
        tokens.append(token)

    assert "".join(tokens) == "AB"


@pytest.mark.asyncio
async def test_workflow_classify_flow_validates_non_empty_categories():
    service = WorkflowService(_FakeRagService(), _FakeAuditService())

    with pytest.raises(ValidationError, match="At least one category"):
        await service.classify_flow("text", [], context=_ctx())


@pytest.mark.asyncio
async def test_workflow_classify_flow_success_confidence_logic():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.complete_text.return_value = SimpleNamespace(raw_text="invoice", model="m", latency_ms=3)

    service = WorkflowService(rag, audit)
    out = await service.classify_flow("text", ["invoice", "letter"], context=_ctx())

    assert out["predicted_category"] == "invoice"
    assert out["confidence_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_workflow_summarize_flow_success_and_key_points_cap():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.complete_text.return_value = SimpleNamespace(
        raw_text="One. Two. Three. Four.", model="m", latency_ms=5
    )

    service = WorkflowService(rag, audit)
    out = await service.summarize_flow("text", max_length=100, context=_ctx())

    assert len(out["key_points"]) == 3
    assert out["summary"].startswith("One")


@pytest.mark.asyncio
async def test_workflow_summarize_flow_records_failure_audit():
    rag = _FakeRagService()
    audit = _FakeAuditService()
    rag.complete_text.side_effect = RuntimeError("llm down")

    service = WorkflowService(rag, audit)

    with pytest.raises(RuntimeError, match="llm down"):
        await service.summarize_flow("text", context=_ctx())

    kwargs = audit.record_flow_execution.await_args.kwargs
    assert kwargs["success"] is False


@pytest.mark.asyncio
async def test_rag_retrieve_documents_uses_chunk_results_when_available(monkeypatch):
    llm_service = SimpleNamespace(complete=AsyncMock(), stream_complete=AsyncMock())
    repo = SimpleNamespace(session=object(), list=AsyncMock())
    search_provider = SimpleNamespace(search=AsyncMock(), get_name=lambda: "mock")
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    repo.list.return_value = [SimpleNamespace(id="d1", title="Doc 1", text="T")]

    class _Pipeline:
        retrieve = AsyncMock(
            return_value=[{"document_id": "d1", "text": "chunk", "score": 0.9, "chunk_index": 0}]
        )

    monkeypatch.setattr("app.services.rag_service.rag_pipeline", _Pipeline)

    results = await rag.retrieve_documents("q", context=_ctx())

    assert results[0]["title"] == "Doc 1"
    assert results[0]["chunk_index"] == 0


@pytest.mark.asyncio
async def test_rag_retrieve_documents_falls_back_to_text_matching(monkeypatch):
    llm_service = SimpleNamespace(complete=AsyncMock(), stream_complete=AsyncMock())
    docs = [
        SimpleNamespace(id="d1", title="Doc 1", text="Paris is capital of France"),
        SimpleNamespace(id="d2", title="Doc 2", text="Completely unrelated"),
    ]
    repo = SimpleNamespace(session=object(), list=AsyncMock(return_value=docs))
    search_provider = SimpleNamespace(search=AsyncMock(), get_name=lambda: "mock")
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    class _Pipeline:
        retrieve = AsyncMock(return_value=[])

    monkeypatch.setattr("app.services.rag_service.rag_pipeline", _Pipeline)

    results = await rag.retrieve_documents("Paris", context=_ctx())

    assert len(results) == 1
    assert results[0]["document_id"] == "d1"


@pytest.mark.asyncio
async def test_rag_answer_question_uses_llm_and_sources():
    llm_service = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(raw_text="answer", model="m", latency_ms=4)
        ),
        stream_complete=AsyncMock(),
    )
    repo = SimpleNamespace(session=object(), list=AsyncMock(return_value=[]))
    search_provider = SimpleNamespace(search=AsyncMock(), get_name=lambda: "mock")
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    docs = [{"document_id": "d1", "title": "Doc 1", "text": "ctx", "score": 0.7}]
    out = await rag.answer_question("q", context_documents=docs, context=_ctx())

    assert out["answer"] == "answer"
    assert out["sources"][0]["document_id"] == "d1"


@pytest.mark.asyncio
async def test_rag_search_external_formats_results():
    llm_service = SimpleNamespace(complete=AsyncMock(), stream_complete=AsyncMock())
    search_results = [SimpleNamespace(title="T", url="u", snippet="s")]
    search_provider = SimpleNamespace(
        search=AsyncMock(return_value=search_results),
        get_name=lambda: "duckduckgo",
    )
    repo = SimpleNamespace(session=object(), list=AsyncMock(return_value=[]))
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    out = await rag.search_external("q", context=_ctx())

    assert out[0]["source"] == "duckduckgo"


@pytest.mark.asyncio
async def test_rag_complete_text_proxies_to_llm():
    llm_service = SimpleNamespace(
        complete=AsyncMock(return_value=SimpleNamespace(raw_text="x", model="m", latency_ms=1)),
        stream_complete=AsyncMock(),
    )
    search_provider = SimpleNamespace(search=AsyncMock(), get_name=lambda: "mock")
    repo = SimpleNamespace(session=object(), list=AsyncMock(return_value=[]))
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    result = await rag.complete_text("prompt", context=_ctx())

    assert result.raw_text == "x"


@pytest.mark.asyncio
async def test_rag_stream_answer_yields_tokens_with_provided_docs():
    async def _stream_complete(prompt, *, system_prompt=None, context=None):
        assert "Context:" in prompt
        yield "a"
        yield "b"

    llm_service = SimpleNamespace(complete=AsyncMock(), stream_complete=_stream_complete)
    search_provider = SimpleNamespace(search=AsyncMock(), get_name=lambda: "mock")
    repo = SimpleNamespace(session=object(), list=AsyncMock(return_value=[]))
    rag = RAGService(
        llm_service,
        embedding_provider=object(),
        search_provider=search_provider,
        document_repository=repo,
    )

    docs = [{"document_id": "d1", "title": "Doc", "text": "ctx", "score": 1.0}]
    tokens = []
    async for t in rag.stream_answer("q", context_documents=docs, context=_ctx()):
        tokens.append(t)

    assert "".join(tokens) == "ab"
