codefrom abc import ABC, abstractmethod
from typing import Any

from banking_investigator.memory.models import MemoryItem


class MemoryStore(ABC):
    """Abstraction for storing and retrieving agent memory."""

    @abstractmethod
    def store(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> MemoryItem:
        """Create or replace a memory item."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        namespace: str,
        key: str,
    ) -> MemoryItem | None:
        """Retrieve one memory item by namespace and key."""
        raise NotImplementedError

    @abstractmethod
    def retrieve_namespace(
        self,
        namespace: str,
    ) -> list[MemoryItem]:
        """Retrieve all memory items within a namespace."""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        namespace: str,
        key: str,
    ) -> None:
        """Delete a memory item."""
        raise NotImplementedError
