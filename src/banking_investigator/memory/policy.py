from enum import StrEnum


class MemoryType(StrEnum):
    """Types of information that may be persisted as long-term memory."""

    CUSTOMER_PREFERENCE = "customer_preference"
    CUSTOMER_PROFILE = "customer_profile"
    INVESTIGATION_CONTEXT = "investigation_context"


ALLOWED_MEMORY_TYPES = {
    MemoryType.CUSTOMER_PREFERENCE,
    MemoryType.CUSTOMER_PROFILE,
    MemoryType.INVESTIGATION_CONTEXT,
}


def is_memory_allowed(memory_type: MemoryType) -> bool:
    """Return whether a memory type is allowed to be persisted."""
    return memory_type in ALLOWED_MEMORY_TYPES
