"""Retrieval-augmented generation service."""

from __future__ import annotations

from typing import Any, Optional

from app.core.context import ExecutionContext, require_execution_context
from app.core.logging import get_logger
from app.persistence.repositories.document import DocumentRepository
from app.providers import EmbeddingProvider, SearchProvider
from app.rag.pipeline import rag_pipeline

from .base_service import BaseService
from .llm_service import LLMService

logger = get_logger(__name__)


class RAGService(BaseService):
    """Service for retrieval-augmented generation.

    Orchestrates document retrieval, embedding, and LLM-based answer generation.
    Provides both standard and streaming completions with source attribution.

    Args:
        llm_service: LLMService for completions
        embedding_provider: Provider for text embeddings
        search_provider: Provider for web search
        document_repository: Repository for document access
    """

    def __init__(
        self,
        llm_service: LLMService,
        embedding_provider: EmbeddingProvider,
        search_provider: SearchProvider,
        document_repository: DocumentRepository,
    ) -> None:
        """Initialize RAG service.

        Args:
            llm_service: LLMService instance
            embedding_provider: EmbeddingProvider instance
            search_provider: SearchProvider instance
            document_repository: DocumentRepository instance
        """
        super().__init__("rag")
        self._llm_service = llm_service
        self._embedding_provider = embedding_provider
        self._search_provider = search_provider
        self._document_repository = document_repository

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        *,
        document_ids: Optional[list[str]] = None,
        context: ExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Maximum documents to retrieve
            document_ids: Optional filter to specific documents
            context: Optional execution context (uses current if not provided)

        Returns:
            List of document chunks with metadata
        """
        ctx = context or require_execution_context()
        self.log_info(
            "rag.retrieve_started",
            query=query[:100],
            top_k=top_k,
            tenant_id=ctx.tenant_id,
        )

        # For now, return mock results. In full implementation, would:
        # 1. Embed the query
        # 2. Search vector database for similar documents
        # 3. Return top-k chunks with scores

        try:
            chunk_results = await rag_pipeline.retrieve(
                tenant_id=str(ctx.tenant_id),
                query=query,
                top_k=top_k,
                document_ids=document_ids,
                db=self._document_repository.session,
            )
            if chunk_results:
                all_docs = await self._document_repository.list(tenant_id=ctx.tenant_id)
                title_by_id = {doc.id: doc.title for doc in all_docs}
                results = [
                    {
                        "document_id": item["document_id"],
                        "title": title_by_id.get(item["document_id"], item["document_id"]),
                        "text": item["text"],
                        "score": float(item.get("score", 0.0)),
                        "chunk_index": item.get("chunk_index"),
                    }
                    for item in chunk_results
                ]
                self.log_info(
                    "rag.retrieve_success",
                    retrieved=len(results),
                    tenant_id=ctx.tenant_id,
                    source="chunks",
                )
                return results

            all_docs = await self._document_repository.list(tenant_id=ctx.tenant_id)

            # Filter by document_ids if provided
            if document_ids:
                all_docs = [d for d in all_docs if d.id in document_ids]

            # Simple text matching (not real semantic search)
            query_lower = query.lower()
            scored_docs = []
            for doc in all_docs:
                score = 0.0
                if query_lower in doc.text.lower():
                    score = 1.0
                elif any(word in doc.text.lower() for word in query_lower.split()):
                    score = 0.5

                if score > 0:
                    scored_docs.append((doc, score))

            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = [
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "score": float(score),
                }
                for doc, score in scored_docs[:top_k]
            ]

            self.log_info(
                "rag.retrieve_success",
                retrieved=len(results),
                tenant_id=ctx.tenant_id,
                source="documents_fallback",
            )
            return results
        except Exception as exc:
            self.log_error(
                "rag.retrieve_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def answer_question(
        self,
        question: str,
        context_documents: Optional[list[dict[str, Any]]] = None,
        *,
        user_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context: ExecutionContext | None = None,
    ) -> dict[str, Any]:
        """Answer a question using retrieved context.

        Args:
            question: Question to answer
            context_documents: Pre-retrieved documents to use as context
            system_prompt: Optional custom system prompt
            context: Optional execution context (uses current if not provided)

        Returns:
            Dict with answer, sources, model name, latency
        """
        ctx = context or require_execution_context()
        self.log_info(
            "rag.answer_question_started",
            question=question[:100],
            num_context_docs=len(context_documents or []),
            tenant_id=ctx.tenant_id,
        )

        # Retrieve documents if not provided
        if not context_documents:
            context_documents = await self.retrieve_documents(question, context=ctx)

        # Build context string from documents
        context_str = (
            "\n\n".join(f"[{doc['document_id']}] {doc['text'][:500]}" for doc in context_documents)
            if context_documents
            else "(No relevant documents found.)"
        )

        # Build prompt
        default_system = (
            "You are a helpful assistant. Answer the question based only on "
            "the provided context. If the context doesn't contain relevant information, say so."
        )
        system = system_prompt or default_system

        question_text = question
        if user_context and user_context.strip():
            question_text = f"{question}\n\nAdditional context:\n{user_context.strip()}"

        user_prompt = (
            "Based on the following context, please answer the question.\n\n"
            f"Context:\n{context_str[:8000]}\n\nQuestion: {question_text}"
        )

        try:
            # Call LLM
            result = await self._llm_service.complete(
                user_prompt,
                system_prompt=system,
                context=ctx,
            )

            answer_dict = {
                "answer": result.raw_text,
                "sources": [
                    {
                        "document_id": doc["document_id"],
                        "title": doc["title"],
                        "score": doc.get("score", 0.0),
                    }
                    for doc in context_documents
                ],
                "model": result.model,
                "latency_ms": result.latency_ms,
            }
            self.log_info(
                "rag.answer_success",
                answer_length=len(result.raw_text),
                source_count=len(context_documents),
                tenant_id=ctx.tenant_id,
            )
            return answer_dict
        except Exception as exc:
            self.log_error(
                "rag.answer_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def search_external(
        self,
        query: str,
        *,
        limit: int = 5,
        context: ExecutionContext | None = None,
    ) -> list[dict[str, Any]]:
        """Search external sources (web search).

        Args:
            query: Search query
            limit: Maximum results to return
            context: Optional execution context (uses current if not provided)

        Returns:
            List of search results with title, url, snippet
        """
        ctx = context or require_execution_context()
        self.log_info(
            "rag.search_external_started",
            query=query[:100],
            limit=limit,
            tenant_id=ctx.tenant_id,
        )

        try:
            results = await self._search_provider.search(query, limit=limit)
            formatted = [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": self._search_provider.get_name(),
                }
                for r in results
            ]
            self.log_info(
                "rag.search_external_success",
                found=len(formatted),
                tenant_id=ctx.tenant_id,
            )
            return formatted
        except Exception as exc:
            self.log_error(
                "rag.search_external_error",
                error=str(exc),
                error_type=type(exc).__name__,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def complete_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        context: ExecutionContext | None = None,
    ):
        """Expose completion via a stable public API for workflow services."""
        ctx = context or require_execution_context()
        return await self._llm_service.complete(prompt, system_prompt=system_prompt, context=ctx)

    async def stream_answer(
        self,
        question: str,
        *,
        context_documents: Optional[list[dict[str, Any]]] = None,
        user_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context: ExecutionContext | None = None,
    ):
        """Stream an answer using retrieved context without exposing private members."""
        ctx = context or require_execution_context()
        documents = context_documents
        if not documents:
            documents = await self.retrieve_documents(question, context=ctx)

        context_str = (
            "\n\n".join(f"[{doc['document_id']}] {doc['text'][:500]}" for doc in documents)
            if documents
            else "(No relevant documents found.)"
        )
        question_text = question
        if user_context and user_context.strip():
            question_text = f"{question}\n\nAdditional context:\n{user_context.strip()}"

        user_prompt = (
            "Based on the following context, please answer the question.\n\n"
            f"Context:\n{context_str[:8000]}\n\nQuestion: {question_text}"
        )
        effective_system = system_prompt or (
            "You are a helpful assistant. Answer the question based only on "
            "the provided context. If the context doesn't contain relevant information, say so."
        )

        async for token in self._llm_service.stream_complete(
            user_prompt,
            system_prompt=effective_system,
            context=ctx,
        ):
            yield token
