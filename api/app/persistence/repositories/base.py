"""Base repository pattern for data access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Abstract repository for data access operations.

    Subclasses implement CRUD operations for specific entity types.
    This pattern enables:
    - Clean separation of persistence logic from domain logic
    - Easy testing with in-memory implementations
    - Swappable storage backends
    """

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create an entity in persistence."""
        pass

    @abstractmethod
    async def read(self, id: Any) -> Optional[T]:
        """Read an entity by ID."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an entity in persistence."""
        pass

    @abstractmethod
    async def delete(self, id: Any) -> bool:
        """Delete an entity by ID. Return True if deleted."""
        pass

    @abstractmethod
    async def list(self, **filters: Any) -> list[T]:
        """List entities matching filters."""
        pass
