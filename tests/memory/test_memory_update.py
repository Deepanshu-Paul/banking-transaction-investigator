from time import sleep
from uuid import uuid4

from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


def test_updating_memory_replaces_value() -> None:
    service = MemoryService()

    namespace = f"customer:{uuid4()}"
    key = "account_profile"

    try:
        first = service.remember(
            namespace=namespace,
            key=key,
            value={
                "account_id": "ACC1001",
                "account_type": "checking",
                "status": "active",
            },
            memory_type=MemoryType.CUSTOMER_PROFILE,
        )

        sleep(0.01)

        second = service.remember(
            namespace=namespace,
            key=key,
            value={
                "account_id": "ACC1001",
                "account_type": "checking",
                "status": "closed",
            },
            memory_type=MemoryType.CUSTOMER_PROFILE,
        )

        assert second.value["status"] == "closed"

        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at

    finally:
        service.forget(namespace, key)


def test_updating_memory_with_ttl_refreshes_expiration() -> None:
    service = MemoryService()

    namespace = f"customer:{uuid4()}"
    key = "temporary_context"

    try:
        first = service.remember(
            namespace=namespace,
            key=key,
            value={"status": "first"},
            memory_type=MemoryType.INVESTIGATION_CONTEXT,
            ttl_seconds=60,
        )

        sleep(0.01)

        second = service.remember(
            namespace=namespace,
            key=key,
            value={"status": "updated"},
            memory_type=MemoryType.INVESTIGATION_CONTEXT,
            ttl_seconds=120,
        )

        assert second.value["status"] == "updated"
        assert second.created_at == first.created_at
        assert second.expires_at is not None
        assert first.expires_at is not None
        assert second.expires_at > first.expires_at

    finally:
        service.forget(namespace, key)
