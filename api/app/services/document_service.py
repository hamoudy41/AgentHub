"""Document management service."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.context import ExecutionContext, require_execution_context
from app.core.errors import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.persistence.models import Document
from app.persistence.repositories.document import DocumentRepository
from app.persistence.schemas import DocumentRead

from .base_service import BaseService

logger = get_logger(__name__)


class DocumentService(BaseService):
    """Service for document lifecycle management.

    Handles CRUD operations and document discovery with tenant scoping.
    Ensures documents are immutable once created (no updates).

    Args:
        repository: DocumentRepository for data access
    """

    def __init__(self, repository: DocumentRepository) -> None:
        """Initialize document service.

        Args:
            repository: DocumentRepository instance
        """
        super().__init__("document")
        self._repository = repository

    async def create(
        self,
        document_id: str,
        title: str,
        text: str,
        *,
        context: ExecutionContext | None = None,
    ) -> DocumentRead:
        """Create a new document.

        Args:
            document_id: Unique document identifier
            title: Document title
            text: Document content
            context: Optional execution context (uses current if not provided)

        Returns:
            Created document with all metadata

        Raises:
            ConflictError: If document_id already exists for this tenant
        """
        ctx = context or require_execution_context()
        self.log_info(
            "document.create_started",
            document_id=document_id,
            title=title[:50],
            text_length=len(text),
            tenant_id=ctx.tenant_id,
            request_id=ctx.request_id,
        )

        try:
            doc = await self._repository.create(
                Document(
                    id=document_id,
                    tenant_id=str(ctx.tenant_id),
                    title=title,
                    text=text,
                )
            )
            self.log_info(
                "document.created",
                document_id=doc.id,
                tenant_id=ctx.tenant_id,
            )
            return self._to_read(doc)
        except ConflictError:
            self.log_warning(
                "document.duplicate_attempt",
                document_id=document_id,
                tenant_id=ctx.tenant_id,
            )
            raise

    async def read(
        self,
        document_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> DocumentRead:
        """Fetch a document by ID.

        Args:
            document_id: Document identifier
            context: Optional execution context (uses current if not provided)

        Returns:
            Document with all metadata

        Raises:
            NotFoundError: If document not found for this tenant
        """
        ctx = context or require_execution_context()
        doc = await self._repository.read(document_id, tenant_id=ctx.tenant_id)
        if not doc:
            self.log_warning(
                "document.not_found",
                document_id=document_id,
                tenant_id=ctx.tenant_id,
            )
            raise NotFoundError(f"Document {document_id} not found for tenant {ctx.tenant_id}")

        self.log_info("document.read", document_id=document_id, tenant_id=ctx.tenant_id)
        return self._to_read(doc)

    async def delete(
        self,
        document_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> None:
        """Delete a document.

        Args:
            document_id: Document identifier
            context: Optional execution context (uses current if not provided)

        Raises:
            NotFoundError: If document not found for this tenant
        """
        ctx = context or require_execution_context()
        doc = await self._repository.read(document_id, tenant_id=ctx.tenant_id)
        if not doc:
            self.log_warning(
                "document.delete_not_found",
                document_id=document_id,
                tenant_id=ctx.tenant_id,
            )
            raise NotFoundError(f"Document {document_id} not found for tenant {ctx.tenant_id}")

        await self._repository.delete(document_id, tenant_id=ctx.tenant_id)
        self.log_info(
            "document.deleted",
            document_id=document_id,
            tenant_id=ctx.tenant_id,
        )

    async def list_for_tenant(
        self,
        *,
        context: ExecutionContext | None = None,
    ) -> list[DocumentRead]:
        """List all documents for current tenant.

        Args:
            context: Optional execution context (uses current if not provided)

        Returns:
            List of documents scoped to tenant
        """
        ctx = context or require_execution_context()
        docs = await self._repository.list(tenant_id=ctx.tenant_id)
        self.log_info(
            "document.list",
            count=len(docs),
            tenant_id=ctx.tenant_id,
        )
        return [self._to_read(doc) for doc in docs]

    @staticmethod
    def _to_read(doc: Document) -> DocumentRead:
        """Convert ORM model to API schema.

        Args:
            doc: Document ORM model

        Returns:
            DocumentRead schema
        """
        return DocumentRead(
            id=doc.id,
            title=doc.title,
            text=doc.text,
            created_at=doc.created_at or datetime.now(timezone.utc),
        )
