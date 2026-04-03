"""Document repository implementation."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.models import Document
from app.core.errors import ConflictError
from app.core.logging import get_logger

from .base import Repository

logger = get_logger(__name__)


class DocumentRepository(Repository[Document]):
    """Repository for document persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, entity: Document) -> Document:
        """Create a new document.

        Args:
            entity: Document to create

        Returns:
            The created document

        Raises:
            ConflictError: If a document with the same ID already exists
        """
        try:
            self.session.add(entity)
            await self.session.commit()
            await self.session.refresh(entity)
            return entity
        except IntegrityError as exc:
            await self.session.rollback()
            logger.warning("document_creation_conflict", document_id=entity.id, error=str(exc))
            error_msg = str(exc).lower()
            if "duplicate" in error_msg or "unique" in error_msg:
                raise ConflictError(f"Document with ID {entity.id} already exists") from exc
            # Re-raise unexpected integrity errors
            raise

    async def read(self, id: str, tenant_id: Optional[str] = None) -> Optional[Document]:
        """Read a document by ID, scoped to tenant."""
        if not tenant_id:
            return None
        result = await self.session.execute(
            select(Document).where(
                Document.id == id,
                Document.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def update(self, entity: Document) -> Document:
        """Update a document."""
        await self.session.merge(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(self, id: str, tenant_id: Optional[str] = None) -> bool:
        """Delete a document by ID, scoped to tenant."""
        if not tenant_id:
            return False
        result = await self.session.execute(
            delete(Document).where(
                Document.id == id,
                Document.tenant_id == tenant_id,
            )
        )
        await self.session.commit()
        return result.rowcount > 0

    async def list(self, tenant_id: str, **filters: dict) -> list[Document]:
        """List documents for a tenant."""
        stmt = select(Document).where(Document.tenant_id == tenant_id)

        if "created_after" in filters:
            stmt = stmt.where(Document.created_at >= filters["created_after"])

        if "title_contains" in filters:
            stmt = stmt.where(Document.title.ilike(f"%{filters['title_contains']}%"))

        result = await self.session.execute(stmt)
        return result.scalars().all()
