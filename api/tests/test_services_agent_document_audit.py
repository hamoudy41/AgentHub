from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.context import ExecutionContext
from app.core.errors import ConflictError, NotFoundError
from app.models import AiCallAudit
from app.services.agent_service import AgentService
from app.services.audit_service import AuditService
from app.services.document_service import DocumentService
from app.services.memory_service import MemoryService
from app.services.tool_service import ToolService


def _ctx(tenant: str = "tenant-1") -> ExecutionContext:
    return ExecutionContext.from_request(tenant)


def _build_agent_service(document_repository: object | None = None) -> AgentService:
    tool_service = ToolService()
    memory_service = MemoryService()
    return AgentService(tool_service, memory_service, document_repository=document_repository)


@pytest.mark.asyncio
async def test_agent_create_get_delete_and_list_by_tenant():
    service = _build_agent_service()

    a1 = service.create_agent("a1", "A One", "m1", context=_ctx("tenant-1"))
    service.create_agent("a2", "A Two", "m2", context=_ctx("tenant-2"))

    assert service.get_agent("a1", context=_ctx("tenant-1")).id == a1.id

    tenant_1_agents = await service.list_agents(context=_ctx("tenant-1"))
    assert [a["id"] for a in tenant_1_agents] == ["a1"]

    service.delete_agent("a1", context=_ctx("tenant-1"))
    with pytest.raises(NotFoundError):
        service.get_agent("a1", context=_ctx("tenant-1"))


def test_agent_get_missing_raises_not_found():
    service = _build_agent_service()
    with pytest.raises(NotFoundError):
        service.get_agent("missing", context=_ctx())


@pytest.mark.asyncio
async def test_agent_add_remove_and_get_tools():
    service = _build_agent_service()
    ctx = _ctx()
    service.create_agent("a1", "A", "m", context=ctx)

    service._tool_service.register_tool("t1", "adder", "desc", {}, lambda a, b: a + b)

    await service.add_tool_to_agent("a1", "t1", context=ctx)
    tools = service.get_agent_tools("a1", context=ctx)
    assert len(tools) == 1
    assert tools[0].definition.id == "t1"

    await service.remove_tool_from_agent("a1", "t1", context=ctx)
    assert service.get_agent_tools("a1", context=ctx) == []


@pytest.mark.asyncio
async def test_agent_execute_success(monkeypatch):
    service = _build_agent_service()
    ctx = _ctx()
    service.create_agent("a1", "A", "test-model", context=ctx)

    async def _fake_run_agent(*, tenant_id: str, message: str, get_document_fn):
        await asyncio.sleep(0)
        assert tenant_id == "tenant-1"
        assert message == "hello"
        return {"answer": "ok", "tools_used": ["search"], "error": None}

    monkeypatch.setattr("app.agents.react_agent.run_agent", _fake_run_agent)

    result = await service.execute_agent("a1", "hello", context=ctx)

    assert result["status"] == "completed"
    assert result["result"] == "ok"
    assert result["tools_used"] == ["search"]
    assert service.get_agent("a1", context=ctx).state.value == "idle"


@pytest.mark.asyncio
async def test_agent_execute_exposes_document_callback_to_runtime(monkeypatch):
    repo = SimpleNamespace(read=AsyncMock())
    created = datetime(2025, 1, 1, tzinfo=timezone.utc)
    repo.read.return_value = SimpleNamespace(
        id="d1",
        title="Doc",
        text="Body",
        created_at=created,
    )
    service = _build_agent_service(document_repository=repo)
    ctx = _ctx()
    service.create_agent("a1", "A", "m", context=ctx)

    async def _fake_run_agent(*, tenant_id: str, message: str, get_document_fn):
        doc = await get_document_fn("d1", tenant_id)
        assert doc == {
            "id": "d1",
            "title": "Doc",
            "text": "Body",
            "created_at": created.isoformat(),
        }
        return {"answer": "done", "tools_used": [], "error": None}

    monkeypatch.setattr("app.agents.react_agent.run_agent", _fake_run_agent)

    out = await service.execute_agent("a1", "q", context=ctx)
    assert out["result"] == "done"
    repo.read.assert_awaited_once_with("d1", tenant_id="tenant-1")


@pytest.mark.asyncio
async def test_agent_execute_failure_sets_stopped(monkeypatch):
    service = _build_agent_service()
    ctx = _ctx()
    service.create_agent("a1", "A", "m", context=ctx)

    async def _fake_run_agent(*, tenant_id: str, message: str, get_document_fn):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.agents.react_agent.run_agent", _fake_run_agent)

    with pytest.raises(RuntimeError, match="boom"):
        await service.execute_agent("a1", "q", context=ctx)

    assert service.get_agent("a1", context=ctx).state.value == "stopped"


@pytest.mark.asyncio
async def test_document_service_create_read_list_delete_roundtrip():
    repo = SimpleNamespace(
        create=AsyncMock(),
        read=AsyncMock(),
        list=AsyncMock(),
        delete=AsyncMock(),
    )
    service = DocumentService(repo)
    ctx = _ctx()

    doc_model = SimpleNamespace(
        id="d1",
        title="Title",
        text="Text",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    repo.create.return_value = doc_model
    repo.read.return_value = doc_model
    repo.list.return_value = [doc_model]

    created = await service.create("d1", "Title", "Text", context=ctx)
    read_back = await service.read("d1", context=ctx)
    all_docs = await service.list_for_tenant(context=ctx)
    await service.delete("d1", context=ctx)

    assert created.id == "d1"
    assert read_back.title == "Title"
    assert len(all_docs) == 1
    repo.delete.assert_awaited_once_with("d1", tenant_id="tenant-1")


@pytest.mark.asyncio
async def test_document_service_conflict_and_not_found_paths():
    repo = SimpleNamespace(
        create=AsyncMock(side_effect=ConflictError("exists")),
        read=AsyncMock(return_value=None),
        delete=AsyncMock(),
        list=AsyncMock(return_value=[]),
    )
    service = DocumentService(repo)
    ctx = _ctx()

    with pytest.raises(ConflictError):
        await service.create("d1", "Title", "Text", context=ctx)

    with pytest.raises(NotFoundError):
        await service.read("missing", context=ctx)

    with pytest.raises(NotFoundError):
        await service.delete("missing", context=ctx)


def test_document_to_read_fills_created_at_when_missing():
    doc = SimpleNamespace(id="d1", title="T", text="X", created_at=None)
    out = DocumentService._to_read(doc)
    assert out.created_at.tzinfo is not None


@pytest.mark.asyncio
async def test_audit_record_flow_execution_returns_created_id():
    repo = SimpleNamespace(create=AsyncMock())
    service = AuditService(repo)
    ctx = _ctx()

    async def _create(log: AiCallAudit):
        await asyncio.sleep(0)
        assert log.flow_name == "ask"
        assert log.tenant_id == "tenant-1"
        return log

    repo.create.side_effect = _create

    record_id = await service.record_flow_execution(
        "ask",
        {"question": "q"},
        {"answer": "a"},
        success=True,
        context=ctx,
    )
    assert isinstance(record_id, str)


@pytest.mark.asyncio
async def test_audit_record_tool_call_noop_path():
    repo = SimpleNamespace()
    service = AuditService(repo)
    await service.record_tool_call(
        "a1",
        "search",
        {"q": "hi"},
        {"ok": True},
        success=True,
        latency_ms=5.2,
        context=_ctx(),
    )


@pytest.mark.asyncio
async def test_audit_purge_old_records_disabled_and_enabled():
    repo = SimpleNamespace(purge_older_than=AsyncMock(return_value=4))
    service = AuditService(repo)
    ctx = _ctx()

    disabled = await service.purge_old_records(0, context=ctx)
    enabled = await service.purge_old_records(30, context=ctx)

    assert disabled == 0
    assert enabled == 4
    assert repo.purge_older_than.await_count == 1


@pytest.mark.asyncio
async def test_audit_get_flow_stats_with_and_without_records():
    now = datetime.now(timezone.utc)
    repo = SimpleNamespace(list=AsyncMock())
    service = AuditService(repo)

    repo.list.return_value = []
    empty = await service.get_flow_stats("ask", days=7, context=_ctx())
    assert empty["total_executions"] == 0
    assert empty["success_rate_percent"] == pytest.approx(0.0)

    repo.list.return_value = [
        SimpleNamespace(success=True, created_at=now),
        SimpleNamespace(success=False, created_at=now),
        SimpleNamespace(success=True, created_at=now),
    ]
    mixed = await service.get_flow_stats("ask", days=7, context=_ctx())
    assert mixed["total_executions"] == 3
    assert mixed["successes"] == 2
    assert mixed["failures"] == 1
    assert mixed["success_rate_percent"] == pytest.approx(66.67)
