from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.metrics import LLM_CALLS
from app.llm.errors import LLMNotConfiguredError
from app.schemas import ClassifyRequest, ClassifyResponse

from .common import (
    AiFlowError,
    LLM_NOT_CONFIGURED_MESSAGE,
    persist_audit_record,
    sanitize_flow_text,
)


logger = get_logger(__name__)


async def run_classify_flow(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: ClassifyRequest,
    llm,
) -> ClassifyResponse:
    if not payload.candidate_labels:
        raise AiFlowError("candidate_labels cannot be empty")

    text = sanitize_flow_text(
        payload.text,
        tenant_id=tenant_id,
        max_length=10000,
        log_event="classify.input_validation_failed",
    )

    labels_str = ", ".join(payload.candidate_labels)
    prompt = (
        "Classify the following text by document type. "
        f"Choose exactly one label from: {labels_str}. "
        "Reply with only the single label word, nothing else.\n\n"
        f"Text:\n{text[:4000]}"
    )

    source = "llm"
    try:
        result = await llm.complete(
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
        labels_lower = [candidate.lower() for candidate in payload.candidate_labels]
        if first_word in labels_lower:
            matched_index = labels_lower.index(first_word)
            label = payload.candidate_labels[matched_index]
        else:
            label = payload.candidate_labels[0]
        LLM_CALLS.labels(flow="classify", source="llm").inc()
        response = ClassifyResponse(
            label=label,
            confidence=0.9,
            model=result.model,
            source=source,
            metadata={"latency_ms": result.latency_ms},
        )
    except LLMNotConfiguredError:
        raise AiFlowError(LLM_NOT_CONFIGURED_MESSAGE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ai_flow.classify_failed", error=str(exc))
        source = "fallback"
        LLM_CALLS.labels(flow="classify", source="fallback").inc()
        response = ClassifyResponse(
            label=payload.candidate_labels[0],
            confidence=0.0,
            model="fallback",
            source=source,
            metadata={"fallback_reason": str(exc)},
        )

    response.metadata["audit_persisted"] = await persist_audit_record(
        db,
        tenant_id=tenant_id,
        flow_name="classify",
        request_payload=payload.model_dump(),
        response_payload=response.model_dump(),
        success=source == "llm",
    )
    return response
