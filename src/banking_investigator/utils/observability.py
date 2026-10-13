import json
from functools import wraps
from time import perf_counter
from typing import Any, Callable

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
    "trace_id",
    "node_name",
    "status",
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


def observe_node(
    node_name: str,
    node_function: Callable[..., Any],
) -> Callable[..., Any]:
    """Log safe start, completion, and failure events for a graph node."""
    if not node_name.strip():
        raise ValueError("Node name must not be empty.")

    @wraps(node_function)
    def wrapped(
        state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        trace_id = (
            state.get("trace_id")
            if isinstance(state, dict)
            else None
        )

        message_count = (
            len(state.get("messages", []))
            if isinstance(state, dict)
            else 0
        )

        started_at = perf_counter()

        log_agent_event(
            "node_started",
            trace_id=trace_id,
            node_name=node_name,
            message_count=message_count,
        )

        try:
            result = node_function(
                state,
                *args,
                **kwargs,
            )
        except Exception as exc:
            log_agent_event(
                "node_failed",
                trace_id=trace_id,
                node_name=node_name,
                duration_ms=round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
                success=False,
                error_type=type(exc).__name__,
            )
            raise

        result_trace_id = trace_id
        result_status = None

        if isinstance(result, dict):
            result_trace_id = result.get(
                "trace_id",
                trace_id,
            )
            result_status = result.get("run_status")

        log_agent_event(
            "node_completed",
            trace_id=result_trace_id,
            node_name=node_name,
            duration_ms=round(
                (perf_counter() - started_at) * 1000,
                2,
            ),
            success=True,
            status=result_status,
        )

        return result

    return wrapped