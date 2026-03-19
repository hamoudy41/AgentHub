from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document
from app.schemas import DocumentRead


LEGACY_CREATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_DOCUMENT_ID_LENGTH = 64
MAX_DOCUMENT_TITLE_LENGTH = 255
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class DocumentConflictError(Exception):
    """Raised when a tenant attempts to create a duplicate document."""


class UploadValidationError(Exception):
    """Raised when uploaded document metadata or contents are invalid."""


class UploadTooLargeError(UploadValidationError):
    """Raised when an uploaded document exceeds the platform limit."""


@dataclass(frozen=True)
class PreparedUpload:
    document_id: str
    title: str
    text: str


def _message_indicates_duplicate(message: str) -> bool:
    lowered = message.lower()
    if any(
        token in lowered for token in ("foreign key", "check constraint", "not null", "notnull")
    ):
        return False
    return any(
        token in lowered
        for token in (
            "unique constraint",
            "unique violation",
            "duplicate key",
            "duplicate entry",
            "already exists",
        )
    )


def _is_duplicate_key_error(exc: BaseException) -> bool:
    if not isinstance(exc, IntegrityError):
        return _message_indicates_duplicate(f"{exc}")

    original = getattr(exc, "orig", None)
    if original is not None:
        sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
        if sqlstate == "23505":
            return True
        if type(original).__name__ == "UniqueViolationError":
            return True
        cause = getattr(original, "__cause__", None)
        if cause is not None and type(cause).__name__ == "UniqueViolationError":
            return True
    return _message_indicates_duplicate(f"{exc}")


def document_to_read(document: Document) -> DocumentRead:
    """Convert a persisted document into the API schema."""
    created_at = document.created_at if document.created_at is not None else LEGACY_CREATED_AT
    return DocumentRead(
        id=document.id,
        title=document.title,
        text=document.text,
        created_at=created_at,
    )


async def fetch_document(
    db: AsyncSession,
    tenant_id: str,
    document_id: str,
) -> Document | None:
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
        )
    )
    return result.scalar_one_or_none()


async def fetch_document_payload(
    db: AsyncSession,
    tenant_id: str,
    document_id: str,
) -> dict[str, str] | None:
    document = await fetch_document(db, tenant_id, document_id)
    if not document:
        return None
    return {"id": document.id, "title": document.title, "text": document.text}


async def create_document(
    db: AsyncSession,
    tenant_id: str,
    *,
    document_id: str,
    title: str,
    text: str,
) -> Document:
    existing = await fetch_document(db, tenant_id, document_id)
    if existing is not None:
        raise DocumentConflictError(document_id)

    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        title=title,
        text=text,
    )
    db.add(document)
    try:
        await db.commit()
        await db.refresh(document)
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        if _is_duplicate_key_error(exc):
            raise DocumentConflictError(document_id) from exc
        raise
    return document


async def prepare_uploaded_document(
    file: UploadFile,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> PreparedUpload:
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        max_megabytes = MAX_UPLOAD_BYTES // 1024 // 1024
        raise UploadTooLargeError(f"File too large (max {max_megabytes} MB)")

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploadValidationError("File could not be decoded as UTF-8 text") from exc

    resolved_document_id = document_id or (Path(file.filename or "upload").stem or "upload")
    resolved_title = title or (file.filename or "Uploaded document")

    if len(resolved_document_id) > MAX_DOCUMENT_ID_LENGTH:
        raise UploadValidationError(
            f"Document ID must be at most {MAX_DOCUMENT_ID_LENGTH} characters. "
            "Provide a shorter document_id or use a shorter filename."
        )

    return PreparedUpload(
        document_id=resolved_document_id,
        title=resolved_title[:MAX_DOCUMENT_TITLE_LENGTH],
        text=text,
    )
