from uuid import uuid4

from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


def test_memory_persists_across_service_instances() -> None:
    namespace = f"customer:CUST{uuid4().hex[:8].upper()}"
    key = "account_profile"

    first_service = MemoryService()

    try:
        first_service.remember(
            namespace=namespace,
            key=key,
            value={
                "account_id": "ACC1001",
                "account_type": "checking",
                "status": "active",
            },
            memory_type=MemoryType.CUSTOMER_PROFILE,
        )

        # Simulate a completely new agent run.
        second_service = MemoryService()

        memories = second_service.recall(namespace)

        assert len(memories) == 1
        assert memories[0].namespace == namespace
        assert memories[0].key == key
        assert memories[0].value == {
            "account_id": "ACC1001",
            "account_type": "checking",
            "status": "active",
        }

    finally:
        first_service.forget(namespace, key)
