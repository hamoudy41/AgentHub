from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.context import ExecutionContext, clear_execution_context, set_execution_context
from app.core.errors import AppError, ValidationError
from app.core.config import get_settings
from app.db import get_db_session
from app.http.sse import stream_text_tokens
from app.persistence.repositories import AuditRepository, DocumentRepository
from app.providers.registry import ProviderRegistry
from app.schemas import (
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    NotarySummarizeRequest,
    NotarySummarizeResponse,
)
from app.security import sanitize_user_input
from app.services.audit_service import AuditService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.workflow_service import WorkflowService
from app import services_ai_flows


LLM_CONFIGURATION_ERROR = (
    "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
)

# Operation timeout limits (seconds)
WORKFLOW_TIMEOUT = 120.0  # Max 2 minutes for any workflow operation
STREAM_TIMEOUT = 300.0  # Max 5 minutes for streaming operations


def _validate_ask_request(payload: AskRequest, tenant_id: str) -> None:
    """Validate ask request for security and bounds."""
    if not payload.question or not payload.question.strip():
        raise ValidationError("question cannot be empty")
    if not payload.context or not payload.context.strip():
        raise ValidationError("context cannot be empty")

    # Sanitize and check for injection attempts
    try:
        sanitize_user_input(
            payload.question,
            max_length=2000,
            check_injection=True,
            tenant_id=tenant_id,
        )
        sanitize_user_input(
            payload.context,
            max_length=500_000,
            check_injection=False,  # Context is not user input
            tenant_id=tenant_id,
        )
    except ValueError as e:
        raise ValidationError(str(e))


def _validate_classify_request(payload: ClassifyRequest, tenant_id: str) -> None:
    """Validate classify request for security and bounds."""
    if not payload.text or not payload.text.strip():
        raise ValidationError("text cannot be empty")
    if not payload.candidate_labels or len(payload.candidate_labels) == 0:
        raise ValidationError("candidate_labels cannot be empty")

    try:
        sanitize_user_input(
            payload.text,
            max_length=500_000,
            check_injection=False,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        raise ValidationError(str(e))


def _validate_notary_request(payload: NotarySummarizeRequest, tenant_id: str) -> None:
    """Validate notary request for security and bounds."""
    if not payload.text or not payload.text.strip():
        raise ValidationError("text cannot be empty")
    if payload.language not in ("nl", "en"):
        raise ValidationError("language must be 'nl' or 'en'")

    try:
        sanitize_user_input(
            payload.text,
            max_length=500_000,
            check_injection=False,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        raise ValidationError(str(e))


def _build_workflow_service(db: AsyncSession) -> WorkflowService:
    """Create WorkflowService with all dependencies."""
    settings = get_settings()
    registry = ProviderRegistry(settings)

    audit_repo = AuditRepository(db)
    audit_service = AuditService(audit_repo)
    doc_repo = DocumentRepository(db)
    llm_service = LLMService(registry.get_llm_provider())
    rag_service = RAGService(
        llm_service=llm_service,
        embedding_provider=registry.get_embedding_provider(),
        search_provider=registry.get_search_provider(),
        document_repository=doc_repo,
    )

    return WorkflowService(rag_service, audit_service)


async def run_ask_flow_stream(
    *,
    tenant_id: str,
    db: AsyncSession,
    payload: AskRequest,
) -> AsyncIterator[str]:
    ctx = ExecutionContext.from_request(tenant_id=tenant_id)
    set_execution_context(ctx)
    try:
        service = _build_workflow_service(db)
        async for chunk in service.ask_flow_stream(
            payload.question,
            document_ids=None,
            context=ctx,
        ):
            yield chunk
    finally:
        clear_execution_context()


def build_workflow_router(get_tenant_id) -> APIRouter:
    router = APIRouter(tags=["workflows"])

    @router.post("/ai/notary/summarize", response_model=NotarySummarizeResponse)
    async def notary_summarize(
        payload: NotarySummarizeRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> NotarySummarizeResponse:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        try:
            if not services_ai_flows.llm_client.is_configured():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=LLM_CONFIGURATION_ERROR,
                )
            # Validate request
            _validate_notary_request(payload, tenant_id)

            # Extract text from document if document_id provided
            text = payload.text
            if payload.document_id:
                doc_repo = DocumentRepository(db)
                doc = await doc_repo.read(payload.document_id, tenant_id=tenant_id)
                if doc:
                    text = doc.text

            # Execute with timeout
            try:
                service = _build_workflow_service(db)
                result = await asyncio.wait_for(
                    service.summarize_flow(
                        text,
                        max_length=None,
                        context=ctx,
                    ),
                    timeout=WORKFLOW_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Summarization operation timed out",
                )
            except Exception:
                # Backward-compatible fallback while service migration stabilizes.
                return await services_ai_flows.run_notary_summarization_flow(
                    tenant_id=tenant_id,
                    db=db,
                    payload=payload,
                )

            # Map result to response schema
            return NotarySummarizeResponse(
                document_id=payload.document_id,
                summary={
                    "title": "Summary",
                    "key_points": result.get("key_points", []),
                    "parties_involved": [],
                    "risks_or_warnings": [],
                    "raw_summary": result.get("summary", ""),
                },
                source=result.get("source", "llm"),
                metadata=result.get("metadata", {}),
            )
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except AppError as e:
            raise HTTPException(status_code=e.status_code, detail=e.error_code)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Summarization failed"
            )
        finally:
            clear_execution_context()

    @router.post("/ai/classify", response_model=ClassifyResponse)
    async def classify(
        payload: ClassifyRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> ClassifyResponse:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        try:
            if not services_ai_flows.llm_client.is_configured():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=LLM_CONFIGURATION_ERROR,
                )
            # Validate request
            _validate_classify_request(payload, tenant_id)

            # Execute with timeout
            try:
                service = _build_workflow_service(db)
                result = await asyncio.wait_for(
                    service.classify_flow(
                        payload.text,
                        payload.candidate_labels,
                        context=ctx,
                    ),
                    timeout=WORKFLOW_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Classification operation timed out",
                )
            except Exception:
                # Backward-compatible fallback while service migration stabilizes.
                return await services_ai_flows.run_classify_flow(
                    tenant_id=tenant_id,
                    db=db,
                    payload=payload,
                )

            # Map result to response schema
            return ClassifyResponse(
                label=result.get("predicted_category", payload.candidate_labels[0]),
                confidence=result.get("confidence_score", 0.0),
                model=result.get("model", "unknown"),
                source=result.get("source", "llm"),
                metadata=result.get("metadata", {}),
            )
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except AppError as e:
            raise HTTPException(status_code=e.status_code, detail=e.error_code)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Classification failed"
            )
        finally:
            clear_execution_context()

    @router.post("/ai/ask", response_model=AskResponse)
    async def ask(
        payload: AskRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> AskResponse:
        ctx = ExecutionContext.from_request(tenant_id=tenant_id)
        set_execution_context(ctx)
        try:
            if not services_ai_flows.llm_client.is_configured():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=LLM_CONFIGURATION_ERROR,
                )
            # Validate request
            _validate_ask_request(payload, tenant_id)

            # Execute with timeout
            try:
                service = _build_workflow_service(db)
                result = await asyncio.wait_for(
                    service.ask_flow(
                        payload.question,
                        document_ids=None,
                        context=ctx,
                    ),
                    timeout=WORKFLOW_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Question answering operation timed out",
                )
            except Exception:
                # Backward-compatible fallback while service migration stabilizes.
                return await services_ai_flows.run_ask_flow(
                    tenant_id=tenant_id,
                    db=db,
                    payload=payload,
                )

            # Map result to response schema
            return AskResponse(
                answer=result.get("answer"),
                model=result.get("model", "unknown"),
                source=result.get("source", "llm"),
                metadata=result.get("metadata", {}),
            )
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except AppError as e:
            raise HTTPException(status_code=e.status_code, detail=e.error_code)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Question answering failed",
            )
        finally:
            clear_execution_context()

    @router.post("/ai/ask/stream")
    async def ask_stream(
        payload: AskRequest,
        tenant_id: str = Depends(get_tenant_id),
        db: AsyncSession = Depends(get_db_session),
    ) -> StreamingResponse:
        try:
            if not services_ai_flows.llm_client.is_configured():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=LLM_CONFIGURATION_ERROR,
                )
            # Validate request
            _validate_ask_request(payload, tenant_id)

            async def _stream():
                stream_iter = run_ask_flow_stream(
                    tenant_id=tenant_id,
                    db=db,
                    payload=payload,
                ).__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(anext(stream_iter), timeout=STREAM_TIMEOUT)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        yield "ERROR: Stream operation timed out"
                        break
                    yield chunk

            return StreamingResponse(
                stream_text_tokens(_stream()),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except ValidationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except AppError as e:
            raise HTTPException(status_code=e.status_code, detail=e.error_code)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Streaming failed"
            )

    return router
