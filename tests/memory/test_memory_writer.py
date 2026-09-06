import pytest

from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.writer import MemoryWriter


def test_remember_customer_fact_creates_customer_namespace() -> None:
    writer = MemoryWriter()

    customer_id = "CUST1001"
    key = "preferred_currency"
    value = {"currency": "INR"}

    try:
        memory = writer.remember_customer_fact(
            customer_id=customer_id,
            key=key,
            value=value,
            memory_type=MemoryType.CUSTOMER_PREFERENCE,
        )

        assert memory.namespace == "customer:CUST1001"
        assert memory.key == key
        assert memory.value == value

    finally:
        writer.memory_service.forget(
            namespace="customer:CUST1001",
            key=key,
        )


def test_remember_customer_fact_rejects_empty_customer_id() -> None:
    writer = MemoryWriter()

    with pytest.raises(ValueError, match="customer_id"):
        writer.remember_customer_fact(
            customer_id="",
            key="preferred_currency",
            value={"currency": "INR"},
            memory_type=MemoryType.CUSTOMER_PREFERENCE,
        )


def test_remember_customer_fact_rejects_empty_key() -> None:
    writer = MemoryWriter()

    with pytest.raises(ValueError, match="memory key"):
        writer.remember_customer_fact(
            customer_id="CUST1001",
            key="",
            value={"currency": "INR"},
            memory_type=MemoryType.CUSTOMER_PREFERENCE,
        )
