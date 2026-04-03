"""Unified AI workflow service."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from app.core.context import ExecutionContext, get_execution_context
from app.core.errors import ValidationError
from app.core.logging import get_logger

from .audit_service import AuditService
from .base_service import BaseService
from .rag_service import RAGService

logger = get_logger(__name__)


class WorkflowService(BaseService):
    """Service for orchestrating AI workflows.

    Unifies ask (question-answering), classify (text classification),
    and notary (document summarization) flows through a common interface.

    Args:
        rag_service: RAGService for retrieval and generation
        audit_service: AuditService for logging and auditing
    """

    def __init__(
        self,
        rag_service: RAGService,
        audit_service: AuditService,
    ) -> None:
        """Initialize workflow service.

        Args:
            rag_service: RAGService instance
            audit_service: AuditService instance
        """
        super().__init__("workflow")
        self._rag_service = rag_service
        self._audit_service = audit_service

    async def ask_flow(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute question-answering workflow.

        Args:
            question: Question to answer
            document_ids: Optional filter to specific documents
            context: Optional execution context (uses current if not provided)

        Returns:
            Dict with answer, sources, metadata
        """
        ctx = context or get_execution_context()
        self.log_info(
            "workflow.ask_started",
            question=question[:100],
            tenant_id=ctx.tenant_id,
        )

        try:
            # Retrieve documents
            documents = await self._rag_service.retrieve_documents(
                question,
                top_k=5,
                document_ids=document_ids,
                context=ctx,
            )

            # Generate answer
            response = await self._rag_service.answer_question(
                question,
                context_documents=documents,
                context=ctx,
            )

            # Record audit
            await self._audit_service.record_flow_execution(
                "ask",
                request={"question": question, "document_ids": document_ids},
                response=response,
                success=True,
                context=ctx,
            )

            self.log_info(
                "workflow.ask_completed",
                question=question[:100],
                tenant_id=ctx.tenant_id,
            )
            return response
        except Exception as exc:
            # Record failure
            await self._audit_service.record_flow_execution(
                "ask",
                request={"question": question, "document_ids": document_ids},
                response={"error": str(exc)},
                success=False,
                context=ctx,
            )
            self.log_error(
                "workflow.ask_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def ask_flow_stream(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        *,
        context: ExecutionContext | None = None,
    ) -> AsyncIterator[str]:
        """Execute question-answering workflow with streaming response.

        Args:
            question: Question to answer
            document_ids: Optional filter to specific documents
            context: Optional execution context (uses current if not provided)

        Yields:
            Answer tokens as they are generated
        """
        ctx = context or get_execution_context()
        self.log_info(
            "workflow.ask_stream_started",
            question=question[:100],
            tenant_id=ctx.tenant_id,
        )

        try:
            # Retrieve documents
            documents = await self._rag_service.retrieve_documents(
                question,
                top_k=5,
                document_ids=document_ids,
                context=ctx,
            )

            # Build context string (same as non-streaming)
            context_str = (
                "\n\n".join(f"[{doc['document_id']}] {doc['text'][:500]}" for doc in documents)
                if documents
                else "(No relevant documents found.)"
            )

            user_prompt = (
                "Based on the following context, please answer the question.\n\n"
                f"Context:\n{context_str[:8000]}\n\nQuestion: {question}"
            )

            system_prompt = (
                "You are a helpful assistant. Answer the question based only on "
                "the provided context. If the context doesn't contain relevant information, say so."
            )

            # Stream response
            async for token in self._rag_service._llm_service.stream_complete(
                user_prompt,
                system_prompt=system_prompt,
                context=ctx,
            ):
                yield token

            self.log_info(
                "workflow.ask_stream_completed",
                question=question[:100],
                tenant_id=ctx.tenant_id,
            )
        except Exception as exc:
            self.log_error(
                "workflow.ask_stream_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def classify_flow(
        self,
        text: str,
        categories: list[str],
        *,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute text classification workflow.

        Args:
            text: Text to classify
            categories: List of possible categories
            context: Optional execution context (uses current if not provided)

        Returns:
            Dict with predicted_category, confidence_score, reasoning
        """
        ctx = context or get_execution_context()
        self.log_info(
            "workflow.classify_started",
            text_length=len(text),
            num_categories=len(categories),
            tenant_id=ctx.tenant_id,
        )

        if not categories:
            raise ValidationError("At least one category is required")

        try:
            categories_str = ", ".join(f'"{cat}"' for cat in categories)
            prompt = (
                f"Classify the following text into one of these categories: {categories_str}\n\n"
                f"Text: {text}\n\n"
                f"Respond with ONLY the chosen category name."
            )

            result = await self._rag_service._llm_service.complete(
                prompt,
                system_prompt="You are a text classifier. Always respond with only the category name.",
                context=ctx,
            )

            predicted = result.raw_text.strip()
            confidence = 1.0 if predicted in categories else 0.5

            response = {
                "predicted_category": predicted,
                "confidence_score": confidence,
                "model": result.model,
                "latency_ms": result.latency_ms,
            }

            # Record audit
            await self._audit_service.record_flow_execution(
                "classify",
                request={"text": text[:200], "categories": categories},
                response=response,
                success=True,
                context=ctx,
            )

            self.log_info(
                "workflow.classify_completed",
                predicted=predicted,
                confidence=confidence,
                tenant_id=ctx.tenant_id,
            )
            return response
        except Exception as exc:
            await self._audit_service.record_flow_execution(
                "classify",
                request={"text": text[:200], "categories": categories},
                response={"error": str(exc)},
                success=False,
                context=ctx,
            )
            self.log_error(
                "workflow.classify_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def summarize_flow(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Execute document summarization workflow.

        Args:
            text: Text to summarize
            max_length: Optional max summary length (words)
            context: Optional execution context (uses current if not provided)

        Returns:
            Dict with summary, key_points, model, latency
        """
        ctx = context or get_execution_context()
        self.log_info(
            "workflow.summarize_started",
            text_length=len(text),
            max_length=max_length,
            tenant_id=ctx.tenant_id,
        )

        try:
            length_constraint = f" Keep it under {max_length} words." if max_length else ""
            prompt = (
                f"Summarize the following text concisely.{length_constraint}\n\n"
                f"Text:\n{text[:3000]}"
            )

            result = await self._rag_service._llm_service.complete(
                prompt,
                system_prompt="You are a professional summarizer. Provide concise, accurate summaries.",
                context=ctx,
            )

            summary = result.raw_text.strip()
            # Simple extraction of key points (sentence splitting)
            sentences = [s.strip() for s in summary.split(".") if s.strip()]
            key_points = sentences[:3]  # First 3 sentences as key points

            response = {
                "summary": summary,
                "key_points": key_points,
                "model": result.model,
                "latency_ms": result.latency_ms,
            }

            # Record audit
            await self._audit_service.record_flow_execution(
                "summarize",
                request={"text_length": len(text), "max_length": max_length},
                response={
                    "summary_length": len(summary),
                    "key_points_count": len(key_points),
                    "model": result.model,
                },
                success=True,
                context=ctx,
            )

            self.log_info(
                "workflow.summarize_completed",
                text_length=len(text),
                summary_length=len(summary),
                tenant_id=ctx.tenant_id,
            )
            return response
        except Exception as exc:
            await self._audit_service.record_flow_execution(
                "summarize",
                request={"text_length": len(text)},
                response={"error": str(exc)},
                success=False,
                context=ctx,
            )
            self.log_error(
                "workflow.summarize_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise
