from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.http.sse import stream_text_tokens
from app.schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    NotarySummarizeRequest,
    NotarySummarizeResponse,
)
from app.services_ai_flows import (
    run_ask_flow,
    run_ask_flow_stream,
    run_classify_flow,
    run_notary_summarization_flow,
)
from app.services_llm import llm_client


LLM_CONFIGURATION_ERROR = (
    "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
)


def build_workflow_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["workflows"])

    @router.post("/ai/notary/summarize", response_model=NotarySummarizeResponse)
    async def notary_summarize(
        payload: NotarySummarizeRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> NotarySummarizeResponse:
        return await run_notary_summarization_flow(tenant_id=tenant_id, db=db, payload=payload)

    @router.post("/ai/classify", response_model=ClassifyResponse)
    async def classify(
        payload: ClassifyRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> ClassifyResponse:
        return await run_classify_flow(tenant_id=tenant_id, db=db, payload=payload)

    @router.post("/ai/ask", response_model=AskResponse)
    async def ask(
        payload: AskRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> AskResponse:
        return await run_ask_flow(tenant_id=tenant_id, db=db, payload=payload)

    @router.post("/ai/ask/stream")
    async def ask_stream(
        payload: AskRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> StreamingResponse:
        if not llm_client.is_configured():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=LLM_CONFIGURATION_ERROR
            )

        return StreamingResponse(
            stream_text_tokens(
                run_ask_flow_stream(tenant_id=tenant_id, db=db, payload=payload),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
