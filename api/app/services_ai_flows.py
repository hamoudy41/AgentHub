from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .core.logging import get_logger
from .core.metrics import LLM_CALLS
from .models import AiCallAudit, Document
from .schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    NotarySummarizeRequest,
    NotarySummarizeResponse,
    NotarySummary,
)
from .security import sanitize_user_input, sanitize_for_logging
from .services_llm import LLMError, LLMNotConfiguredError, llm_client


logger = get_logger(__name__)


class AiFlowError(Exception):
    pass


async def run_notary_summarization_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: NotarySummarizeRequest,
) -> NotarySummarizeResponse:
    text = payload.text
    if payload.document_id:
        stmt = select(Document).where(
            Document.id == payload.document_id,
            Document.tenant_id == tenant_id,
        )
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
        if document:
            text = document.text

    # Sanitize input for security
    try:
        text = sanitize_user_input(
            text,
            max_length=50000,  # Reasonable limit for documents
            check_injection=True,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        logger.warning(
            "notary.input_validation_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise AiFlowError(f"Input validation failed: {e}") from e

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

    ai_audit_id = str(uuid.uuid4())

    source: str
    raw_summary: str
    metadata: dict[str, Any] = {}

    try:
        llm_result = await llm_client.generate_notary_summary(prompt, tenant_id=tenant_id)
        raw_summary = llm_result.raw_text
        source = "llm"
        LLM_CALLS.labels(flow="notary_summarize", source="llm").inc()
        metadata.update(
            {
                "model": llm_result.model,
                "latency_ms": llm_result.latency_ms,
            }
        )
    except LLMNotConfiguredError:
        raise AiFlowError(
            "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001
        logger.warning("ai_flow.notary_summarize_llm_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="notary_summarize", source="fallback").inc()
        raw_summary = (
            "Automatische samenvatting niet beschikbaar. "
            "Dit is een veilige, generieke samenvatting op basis van de aangeleverde tekst. "
            "Controleer handmatig de inhoud van de akte."
        )
        metadata.update({"fallback_reason": str(exc)})

    summary = NotarySummary(
        title="Samenvatting notariële akte",
        key_points=[raw_summary[:200]],
        parties_involved=[],
        risks_or_warnings=[],
        raw_summary=raw_summary,
    )

    response = NotarySummarizeResponse(
        document_id=payload.document_id,
        summary=summary,
        source=source,
        metadata=metadata,
    )

    audit = AiCallAudit(
        id=ai_audit_id,
        tenant_id=tenant_id,
        flow_name="notary_summarize",
        request_payload=payload.model_dump(),
        response_payload=response.model_dump(),
        success=source == "llm",
    )
    try:
        db.add(audit)
        await db.commit()
        response.metadata["audit_persisted"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.audit_persist_failed", flow="notary_summarize", error=str(exc))
        try:
            await db.rollback()
        except Exception as rollback_exc:  # noqa: BLE001
            logger.warning(
                "ai_flow.rollback_failed", flow="notary_summarize", error=str(rollback_exc)
            )
        response.metadata["audit_persisted"] = False

    return response


def _audit_flow(
    db: AsyncSession,
    tenant_id: str,
    flow_name: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    success: bool,
) -> bool:
    audit = AiCallAudit(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        flow_name=flow_name,
        request_payload=request_payload,
        response_payload=response_payload,
        success=success,
    )
    try:
        db.add(audit)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.audit_persist_failed", flow=flow_name, error=str(exc))
        return False


async def run_classify_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: ClassifyRequest,
) -> ClassifyResponse:
    if not payload.candidate_labels:
        raise AiFlowError("candidate_labels cannot be empty")

    # Sanitize user input
    try:
        text = sanitize_user_input(
            payload.text,
            max_length=10000,
            check_injection=True,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        logger.warning(
            "classify.input_validation_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise AiFlowError(f"Input validation failed: {e}") from e

    labels_str = ", ".join(payload.candidate_labels)
    prompt = (
        f"Classify the following text by document type. "
        f"Choose exactly one label from: {labels_str}. "
        "Reply with only the single label word, nothing else.\n\nText:\n"
        f"{text[:4000]}"
    )
    source = "llm"
    try:
        result = await llm_client.complete(
            prompt,
            system_prompt=(
                "You are a document classifier. "
                "Classify the given text into one of the provided document-type labels. "
                "Output only the exact label word, nothing else."
            ),
            tenant_id=tenant_id,
        )
        raw = (result.raw_text or "").strip().lower()
        first_word = raw.split()[0] if raw else "other"
        labels_lower = [c.lower() for c in payload.candidate_labels]
        if first_word in labels_lower:
            idx = labels_lower.index(first_word)
            label = payload.candidate_labels[idx]
        else:
            label = payload.candidate_labels[0]
        LLM_CALLS.labels(flow="classify", source="llm").inc()
        out = ClassifyResponse(
            label=label,
            confidence=0.9,
            model=result.model,
            source=source,
            metadata={"latency_ms": result.latency_ms},
        )
    except LLMNotConfiguredError:
        raise AiFlowError(
            "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.classify_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="classify", source="fallback").inc()
        fallback_label = payload.candidate_labels[0] if payload.candidate_labels else "other"
        out = ClassifyResponse(
            label=fallback_label,
            confidence=0.0,
            model="fallback",
            source=source,
            metadata={"fallback_reason": str(exc)},
        )
    audit_added = _audit_flow(
        db,
        tenant_id,
        "classify",
        payload.model_dump(),
        out.model_dump(),
        success=source == "llm",
    )
    if audit_added:
        try:
            await db.commit()
            out.metadata["audit_persisted"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai_flow.audit_persist_failed", flow="classify", error=str(exc))
            try:
                await db.rollback()
            except Exception as rollback_exc:  # noqa: BLE001
                logger.warning("ai_flow.rollback_failed", flow="classify", error=str(rollback_exc))
            out.metadata["audit_persisted"] = False
    else:
        try:
            await db.rollback()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai_flow.rollback_failed", flow="classify", error=str(exc))
        out.metadata["audit_persisted"] = False
    return out


async def run_ask_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
) -> AskResponse:
    # Sanitize user inputs
    try:
        question = sanitize_user_input(
            payload.question,
            max_length=4000,
            check_injection=True,
            tenant_id=tenant_id,
        )
        context = sanitize_user_input(
            payload.context,
            max_length=20000,
            check_injection=True,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        logger.warning(
            "ask.input_validation_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise AiFlowError(f"Input validation failed: {e}") from e

    prompt = (
        "Answer the question based only on the following context. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{context[:8000]}\n\nQuestion: {question}"
    )
    source = "llm"
    try:
        result = await llm_client.complete(
            prompt,
            system_prompt="You are a helpful assistant. Answer concisely based only on the given context.",
            tenant_id=tenant_id,
        )
        out = AskResponse(
            answer=result.raw_text,
            model=result.model,
            source=source,
            metadata={"latency_ms": result.latency_ms},
        )
    except LLMNotConfiguredError:
        raise AiFlowError(
            "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.ask_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="ask", source="fallback").inc()
        out = AskResponse(
            answer="Answer unavailable (model error).",
            model="fallback",
            source=source,
            metadata={"fallback_reason": str(exc)},
        )
    audit_added = _audit_flow(
        db,
        tenant_id,
        "ask",
        payload.model_dump(),
        out.model_dump(),
        success=source == "llm",
    )
    if audit_added:
        try:
            await db.commit()
            out.metadata["audit_persisted"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai_flow.audit_persist_failed", flow="ask", error=str(exc))
            try:
                await db.rollback()
            except Exception as rollback_exc:  # noqa: BLE001
                logger.warning("ai_flow.rollback_failed", flow="ask", error=str(rollback_exc))
            out.metadata["audit_persisted"] = False
    else:
        try:
            await db.rollback()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ai_flow.rollback_failed", flow="ask", error=str(exc))
        out.metadata["audit_persisted"] = False
    return out


async def run_ask_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
) -> AsyncIterator[str]:
    """Stream LLM tokens for the ask flow. Raises AiFlowError if LLM not configured."""
    if not llm_client.is_configured():
        raise AiFlowError(
            "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
        )
    prompt = (
        "Answer the question based only on the following context. "
        "If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{payload.context[:8000]}\n\nQuestion: {payload.question}"
    )
    try:
        async for chunk in llm_client.stream_complete(
            prompt,
            system_prompt="You are a helpful assistant. Answer concisely based only on the given context.",
            tenant_id=tenant_id,
        ):
            yield chunk
        LLM_CALLS.labels(flow="ask_stream", source="llm").inc()
    except LLMNotConfiguredError:
        raise AiFlowError(
            "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.ask_stream_failed", error=str(exc))
        LLM_CALLS.labels(flow="ask_stream", source="fallback").inc()
        yield "Answer unavailable (model error)."
