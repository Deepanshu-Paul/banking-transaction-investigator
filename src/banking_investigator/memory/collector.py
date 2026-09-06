from typing import Any

from banking_investigator.memory.models import MemoryItem
from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


class MemoryCollector:
    """Collect verified information from tool results into long-term memory."""

    def __init__(
        self,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.memory_service = memory_service or MemoryService()

    def collect_account_profile(
        self,
        account_data: dict[str, Any],
    ) -> MemoryItem | None:
        """Store verified account information as customer profile memory."""

        customer_id = account_data.get("customer_id")

        if not customer_id:
            return None

        value = {
            "account_id": account_data.get("account_id"),
            "account_type": account_data.get("account_type"),
            "status": account_data.get("status"),
        }

        return self.memory_service.remember(
            namespace=f"customer:{customer_id}",
            key="account_profile",
            value=value,
            memory_type=MemoryType.CUSTOMER_PROFILE,
        )
