import pytest

from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


def test_remember_stores_allowed_memory() -> None:
    service = MemoryService()

    namespace = "test:customer"
    key = "preferred_currency"
    value = {"currency": "INR"}

    try:
        memory = service.remember(
            namespace=namespace,
            key=key,
            value=value,
            memory_type=MemoryType.CUSTOMER_PREFERENCE,
        )

        assert memory.namespace == namespace
        assert memory.key == key
        assert memory.value == value

        recalled = service.recall(namespace)

        assert len(recalled) == 1
        assert recalled[0].value == value

    finally:
        service.forget(namespace, key)


def test_remember_rejects_invalid_memory_type() -> None:
    service = MemoryService()

    with pytest.raises(ValueError, match="not allowed"):
        service.remember(
            namespace="test:customer",
            key="invalid",
            value={"data": "should not be stored"},
            memory_type="invalid_type",
        )
