from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import run_agent, run_agent_stream
from app.db import get_db_session
from app.documents import fetch_document_payload
from app.http.sse import stream_text_tokens
from app.schemas import AgentChatRequest, AgentChatResponse


def build_agent_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["agents"])

    @router.post("/ai/agents/chat", response_model=AgentChatResponse)
    async def agent_chat(
        payload: AgentChatRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> AgentChatResponse:
        async def get_document(document_id: str, request_tenant_id: str) -> dict | None:
            return await fetch_document_payload(db, request_tenant_id, document_id)

        result = await run_agent(
            tenant_id=tenant_id,
            message=payload.message,
            get_document_fn=get_document,
        )
        return AgentChatResponse(**result)

    @router.post("/ai/agents/chat/stream")
    async def agent_chat_stream(
        payload: AgentChatRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> StreamingResponse:
        async def get_document(document_id: str, request_tenant_id: str) -> dict | None:
            return await fetch_document_payload(db, request_tenant_id, document_id)

        return StreamingResponse(
            stream_text_tokens(
                run_agent_stream(
                    tenant_id=tenant_id,
                    message=payload.message,
                    get_document_fn=get_document,
                ),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
