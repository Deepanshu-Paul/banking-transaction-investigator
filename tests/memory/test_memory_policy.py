import pytest

from banking_investigator.memory.policy import (
    ALLOWED_MEMORY_TYPES,
    MemoryType,
    is_memory_allowed,
)


def test_allowed_memory_types_are_accepted() -> None:
    for memory_type in ALLOWED_MEMORY_TYPES:
        assert is_memory_allowed(memory_type) is True


def test_all_defined_memory_types_are_allowed() -> None:
    for memory_type in MemoryType:
        assert is_memory_allowed(memory_type) is True


def test_invalid_memory_type_is_rejected() -> None:
    assert is_memory_allowed("invalid_type") is False


def test_memory_type_values_are_stable() -> None:
    assert MemoryType.CUSTOMER_PREFERENCE.value == "customer_preference"
    assert MemoryType.CUSTOMER_PROFILE.value == "customer_profile"
    assert MemoryType.INVESTIGATION_CONTEXT.value == "investigation_context"
