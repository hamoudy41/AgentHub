"""Repository module: data access implementations."""

from .base import Repository
from .document import DocumentRepository
from .audit import AuditRepository

__all__ = [
    "Repository",
    "DocumentRepository",
    "AuditRepository",
]
