import random
import time

from banking_investigator.agents.tool_registry import TOOL_REGISTRY
from banking_investigator.models.tool_result import ToolResult
from banking_investigator.services.errors import RetryableError
from banking_investigator.utils.deadline import Deadline

MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 1
MAX_TOTAL_SECONDS = 6


def execute_tool(
    tool_name: str,
    arguments: dict,
    deadline: Deadline | None = None,
) -> ToolResult:

    if deadline is None:
        deadline = Deadline(MAX_TOTAL_SECONDS)

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):

        # Check overall deadline before starting another attempt.
        if deadline.expired():
            return ToolResult(
                success=False,
                error=(
                    f"Tool deadline exceeded after "
                    f"{MAX_TOTAL_SECONDS} seconds"
                ),
            )

        try:
            result = tool(**arguments)

            return ToolResult(
                success=True,
                data=result,
            )

        except RetryableError as exc:
            last_error = exc

            # No more retries allowed.
            if attempt == MAX_RETRIES:
                break

            # Exponential backoff.
            base_backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)

            # Add random jitter to avoid synchronized retries.
            jitter = random.uniform(0, 0.5)

            backoff = base_backoff + jitter

            # Check how much of the overall deadline remains.
            remaining = deadline.remaining_seconds()

            if remaining <= 0:
                break

            # Never sleep longer than the remaining deadline.
            sleep_time = min(backoff, remaining)

            print(
                f"Retrying tool '{tool_name}' "
                f"after {sleep_time:.2f}s..."
            )

            time.sleep(sleep_time)

        except Exception as exc:
            # Permanent / unknown errors fail immediately.
            return ToolResult(
                success=False,
                error=f"Non-retryable tool failure: {exc}",
            )

    return ToolResult(
        success=False,
        error=(
            f"Retryable tool failed after "
            f"{MAX_RETRIES + 1} attempts: {last_error}"
        ),
    )