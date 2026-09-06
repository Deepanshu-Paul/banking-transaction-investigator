from typing import Any

from banking_investigator.memory.models import MemoryItem
from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


class MemoryWriter:
    """Controlled interface for writing verified information to memory."""

    def __init__(
        self,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.memory_service = memory_service or MemoryService()

    def remember_customer_fact(
        self,
        customer_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType,
    ) -> MemoryItem:
        """Store a verified fact associated with a customer."""

        if not customer_id:
            raise ValueError("customer_id must not be empty")

        if not key:
            raise ValueError("memory key must not be empty")

        namespace = f"customer:{customer_id}"

        return self.memory_service.remember(
            namespace=namespace,
            key=key,
            value=value,
            memory_type=memory_type,
        )
