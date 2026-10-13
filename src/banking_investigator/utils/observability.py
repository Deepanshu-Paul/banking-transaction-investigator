import json
from typing import Any

from banking_investigator.config.logging import get_logger


LOGGER_NAME = "banking_investigator.agent"

logger = get_logger(LOGGER_NAME)

SAFE_FIELDS = {
    "allowed_tools",
    "tool_names",
    "tool_name",
    "tool_count",
    "duration_ms",
    "success",
    "error_type",
    "next_agent",
    "message_count",
    "attempt",
}


def log_agent_event(
    event: str,
    **fields: Any,
) -> None:
    """Emit structured events without sensitive banking data."""
    if not event.strip():
        raise ValueError("Event name must not be empty.")

    payload = {
        "event": event,
    }

    payload.update(
        {
            key: value
            for key, value in fields.items()
            if key in SAFE_FIELDS
        }
    )

    logger.info(
        "%s",
        json.dumps(
            payload,
            sort_keys=True,
            default=str,
        ),
    )