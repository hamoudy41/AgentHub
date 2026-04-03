from typing import AsyncIterator
from typing_extensions import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config as core_config
from app.core.context import ExecutionContext, clear_execution_context, set_execution_context
from app.db import get_db_session
from app.http.sse import stream_text_tokens
from app.schemas import AgentChatRequest, AgentChatResponse
from app.services.agent_service import AgentService
from app.services.memory_service import MemoryService
from app.services.tool_service import ToolService


def _build_agent_service() -> AgentService:
    return AgentService(
        tool_service=ToolService(),
        memory_service=MemoryService(),
    )


async def run_agent(*, tenant_id: str, message: str) -> dict[str, object]:
    settings = core_config.get_settings()
    if not settings.llm_provider or not settings.llm_base_url:
        return {
            "answer": "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL.",
            "tools_used": [],
            "error": "llm_not_configured",
        }

    ctx = ExecutionContext.from_request(tenant_id=tenant_id)
    set_execution_context(ctx)
    service = _build_agent_service()
    agent_id = f"default-{tenant_id}"

    try:
        try:
            service.get_agent(agent_id, context=ctx)
        except Exception:
            service.create_agent(
                agent_id=agent_id,
                name="DefaultAgent",
                model="default",
                context=ctx,
            )

        result = await service.execute_agent(agent_id, message, context=ctx)
        return {
            "answer": result.get("result", ""),
            "tools_used": result.get("tools_used", []),
            "error": None,
        }
    finally:
        clear_execution_context()


async def run_agent_stream(*, tenant_id: str, message: str) -> AsyncIterator[str]:
    result = await run_agent(tenant_id=tenant_id, message=message)
    text = str(result.get("answer", ""))
    for token in text.split(" "):
        if token:
            yield f"{token} "


def build_agent_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["agents"])

    @router.post("/ai/agents/chat")
    async def agent_chat(
        payload: AgentChatRequest,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> AgentChatResponse:
        _ = db
        result = await run_agent(tenant_id=tenant_id, message=payload.message)
        return AgentChatResponse(
            answer=str(result.get("answer", "")),
            tools_used=list(result.get("tools_used", [])),
            error=result.get("error"),
        )

    @router.post("/ai/agents/chat/stream")
    async def agent_chat_stream(
        payload: AgentChatRequest,
        tenant_id: Annotated[str, Depends(get_tenant_id)],
        db: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> StreamingResponse:
        _ = db

        async def _stream() -> AsyncIterator[str]:
            async for token in run_agent_stream(
                tenant_id=tenant_id,
                message=payload.message,
            ):
                yield token

        return StreamingResponse(
            stream_text_tokens(
                _stream(),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
