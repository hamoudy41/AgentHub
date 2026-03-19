from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from .flows.ask import run_ask_flow as _run_ask_flow
from .flows.ask import run_ask_flow_stream as _run_ask_flow_stream
from .flows.classify import run_classify_flow as _run_classify_flow
from .flows.common import AiFlowError
from .flows.notary import run_notary_summarization_flow as _run_notary_summarization_flow
from .schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    NotarySummarizeRequest,
    NotarySummarizeResponse,
)
from .services_llm import llm_client

__all__ = [
    "AiFlowError",
    "run_ask_flow",
    "run_ask_flow_stream",
    "run_classify_flow",
    "run_notary_summarization_flow",
]


async def run_notary_summarization_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: NotarySummarizeRequest,
) -> NotarySummarizeResponse:
    return await _run_notary_summarization_flow(
        tenant_id=tenant_id,
        db=db,
        payload=payload,
        llm=llm_client,
    )


async def run_classify_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: ClassifyRequest,
) -> ClassifyResponse:
    return await _run_classify_flow(
        tenant_id=tenant_id,
        db=db,
        payload=payload,
        llm=llm_client,
    )


async def run_ask_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
) -> AskResponse:
    return await _run_ask_flow(
        tenant_id=tenant_id,
        db=db,
        payload=payload,
        llm=llm_client,
    )


async def run_ask_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
) -> AsyncIterator[str]:
    async for chunk in _run_ask_flow_stream(
        tenant_id=tenant_id,
        db=db,
        payload=payload,
        llm=llm_client,
    ):
        yield chunk
