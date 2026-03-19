from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.metrics import LLM_CALLS
from app.llm.errors import LLMNotConfiguredError
from app.schemas import AskRequest, AskResponse

from .common import (
    AiFlowError,
    LLM_NOT_CONFIGURED_MESSAGE,
    persist_audit_record,
    sanitize_flow_text,
)


logger = get_logger(__name__)

ASK_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely based only on the given context."


def build_ask_prompt(question: str, context: str) -> str:
    return (
        "Answer the question based only on the following context. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{context[:8000]}\n\nQuestion: {question}"
    )


async def run_ask_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
    llm,
) -> AskResponse:
    question = sanitize_flow_text(
        payload.question,
        tenant_id=tenant_id,
        max_length=4000,
        log_event="ask.input_validation_failed",
    )
    context = sanitize_flow_text(
        payload.context,
        tenant_id=tenant_id,
        max_length=20000,
        log_event="ask.input_validation_failed",
    )

    prompt = build_ask_prompt(question, context)
    source = "llm"
    try:
        result = await llm.complete(
            prompt,
            system_prompt=ASK_SYSTEM_PROMPT,
            tenant_id=tenant_id,
        )
        response = AskResponse(
            answer=result.raw_text,
            model=result.model,
            source=source,
            metadata={"latency_ms": result.latency_ms},
        )
    except LLMNotConfiguredError:
        raise AiFlowError(LLM_NOT_CONFIGURED_MESSAGE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.ask_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="ask", source="fallback").inc()
        response = AskResponse(
            answer="Answer unavailable (model error).",
            model="fallback",
            source=source,
            metadata={"fallback_reason": str(exc)},
        )

    response.metadata["audit_persisted"] = await persist_audit_record(
        db,
        tenant_id=tenant_id,
        flow_name="ask",
        request_payload=payload.model_dump(),
        response_payload=response.model_dump(),
        success=source == "llm",
    )
    return response


async def run_ask_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
    llm,
) -> AsyncIterator[str]:
    del db

    if not llm.is_configured():
        raise AiFlowError(LLM_NOT_CONFIGURED_MESSAGE)

    prompt = build_ask_prompt(payload.question, payload.context)
    try:
        async for chunk in llm.stream_complete(
            prompt,
            system_prompt=ASK_SYSTEM_PROMPT,
            tenant_id=tenant_id,
        ):
            yield chunk
        LLM_CALLS.labels(flow="ask_stream", source="llm").inc()
    except LLMNotConfiguredError:
        raise AiFlowError(LLM_NOT_CONFIGURED_MESSAGE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.ask_stream_failed", error=str(exc))
        LLM_CALLS.labels(flow="ask_stream", source="fallback").inc()
        yield "Answer unavailable (model error)."
