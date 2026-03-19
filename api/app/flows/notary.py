from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.metrics import LLM_CALLS
from app.documents import fetch_document
from app.llm.errors import LLMError, LLMNotConfiguredError
from app.schemas import NotarySummarizeRequest, NotarySummarizeResponse, NotarySummary

from .common import (
    AiFlowError,
    LLM_NOT_CONFIGURED_MESSAGE,
    persist_audit_record,
    sanitize_flow_text,
)


logger = get_logger(__name__)


async def run_notary_summarization_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: NotarySummarizeRequest,
    llm,
) -> NotarySummarizeResponse:
    text = payload.text
    if payload.document_id:
        document = await fetch_document(db, tenant_id, payload.document_id)
        if document:
            text = document.text

    text = sanitize_flow_text(
        text,
        tenant_id=tenant_id,
        max_length=50000,
        log_event="notary.input_validation_failed",
    )

    prompt = (
        "You are an assistant for Dutch notarial offices. "
        "Summarize the following document in a structured, neutral way. "
        "Only summarize; do not give legal advice or speculate. "
        "Output MUST contain: title; bullet points of key points; parties involved; "
        "any explicit risks or warnings mentioned.\n\n"
        f"LANGUAGE: {payload.language.upper()}\n"
        "DOCUMENT:\n"
        f"{text}"
    )

    source: str
    raw_summary: str
    metadata: dict[str, Any] = {}

    try:
        llm_result = await llm.generate_notary_summary(prompt, tenant_id=tenant_id)
        raw_summary = llm_result.raw_text
        source = "llm"
        LLM_CALLS.labels(flow="notary_summarize", source="llm").inc()
        metadata.update({"model": llm_result.model, "latency_ms": llm_result.latency_ms})
    except LLMNotConfiguredError:
        raise AiFlowError(LLM_NOT_CONFIGURED_MESSAGE)
    except (LLMError, Exception) as exc:  # noqa: BLE001
        logger.warning("ai_flow.notary_summarize_llm_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="notary_summarize", source="fallback").inc()
        raw_summary = (
            "Automatische samenvatting niet beschikbaar. "
            "Dit is een veilige, generieke samenvatting op basis van de aangeleverde tekst. "
            "Controleer handmatig de inhoud van de akte."
        )
        metadata["fallback_reason"] = str(exc)

    response = NotarySummarizeResponse(
        document_id=payload.document_id,
        summary=NotarySummary(
            title="Samenvatting notariële akte",
            key_points=[raw_summary[:200]],
            parties_involved=[],
            risks_or_warnings=[],
            raw_summary=raw_summary,
        ),
        source=source,
        metadata=metadata,
    )
    response.metadata["audit_persisted"] = await persist_audit_record(
        db,
        tenant_id=tenant_id,
        flow_name="notary_summarize",
        request_payload=payload.model_dump(),
        response_payload=response.model_dump(),
        success=source == "llm",
    )
    return response
