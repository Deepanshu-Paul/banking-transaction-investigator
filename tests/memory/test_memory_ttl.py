
import time
from uuid import uuid4

from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


def test_memory_is_available_before_expiration() -> None:
    service = MemoryService()

    namespace = f"test:{uuid4()}"
    key = "temporary_context"
    value = {"status": "active"}

    try:
        service.remember(
            namespace=namespace,
            key=key,
            value=value,
            memory_type=MemoryType.INVESTIGATION_CONTEXT,
            ttl_seconds=5,
        )

        memories = service.recall(namespace)

        assert len(memories) == 1
        assert memories[0].value == value
        assert memories[0].expires_at is not None

    finally:
        service.forget(namespace, key)


def test_expired_memory_is_not_retrieved() -> None:
    service = MemoryService()

    namespace = f"test:{uuid4()}"
    key = "temporary_context"

    try:
        service.remember(
            namespace=namespace,
            key=key,
            value={"status": "temporary"},
            memory_type=MemoryType.INVESTIGATION_CONTEXT,
            ttl_seconds=1,
        )

        time.sleep(1.2)

        result = service.recall(namespace)

        assert result == []

    finally:
        service.forget(namespace, key)


def test_invalid_ttl_is_rejected() -> None:
    service = MemoryService()

    try:
        service.remember(
            namespace="test:customer",
            key="invalid_ttl",
            value={"data": "test"},
            memory_type=MemoryType.INVESTIGATION_CONTEXT,
            ttl_seconds=0,
        )

    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid TTL")
