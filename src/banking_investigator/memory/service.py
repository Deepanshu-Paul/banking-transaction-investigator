from typing import Any

from banking_investigator.memory.models import MemoryItem
from banking_investigator.memory.policy import (
    MemoryType,
    is_memory_allowed,
)
from banking_investigator.memory.postgres_store import PostgresMemoryStore


class MemoryService:
    """Application-level service for managing agent memory."""

    def __init__(
        self,
        store: PostgresMemoryStore | None = None,
    ) -> None:
        self.store = store or PostgresMemoryStore()

    def remember(
        self,
        namespace: str,
        key: str,
        value: Any,
        memory_type: MemoryType,
    ) -> MemoryItem:
        """Persist a memory only when its type is allowed."""

        if not is_memory_allowed(memory_type):
            raise ValueError(
                f"Memory type '{memory_type}' is not allowed."
            )

        return self.store.store(
            namespace=namespace,
            key=key,
            value=value,
        )

    def recall(
        self,
        namespace: str,
    ) -> list[MemoryItem]:
        """Retrieve all memories for a namespace."""

        return self.store.retrieve_namespace(
            namespace=namespace,
        )

    def forget(
        self,
        namespace: str,
        key: str,
    ) -> None:
        """Remove a memory."""

        self.store.delete(
            namespace=namespace,
            key=key,
        )
