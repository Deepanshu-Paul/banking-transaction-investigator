import json
from time import perf_counter

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.config.settings import settings
from banking_investigator.llm_client import client
from banking_investigator.tools.schema import (
    GET_ACCOUNT_TOOL,
    GET_TRANSACTION_TOOL,
)
from banking_investigator.utils.observability import log_agent_event
from banking_investigator.utils.serialization import serialize_for_llm


TOOLS = [
    GET_ACCOUNT_TOOL,
    GET_TRANSACTION_TOOL,
]

MAX_ITERATIONS = 5


def run_agent(user_query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": user_query,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        log_agent_event(
            "legacy_agent_iteration_started",
            attempt=iteration,
        )

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            log_agent_event(
                "legacy_agent_completed",
                attempt=iteration,
                success=True,
            )
            return message.content or ""

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(
                tool_call.function.arguments
            )

            started_at = perf_counter()

            log_agent_event(
                "tool_call_started",
                tool_name=tool_name,
                attempt=iteration,
            )

            try:
                tool_result = execute_tool(
                    tool_name,
                    arguments,
                )
            except Exception as exc:
                log_agent_event(
                    "tool_call_failed",
                    tool_name=tool_name,
                    attempt=iteration,
                    duration_ms=round(
                        (perf_counter() - started_at) * 1000,
                        2,
                    ),
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise

            log_agent_event(
                "tool_call_completed",
                tool_name=tool_name,
                attempt=iteration,
                duration_ms=round(
                    (perf_counter() - started_at) * 1000,
                    2,
                ),
                success=tool_result.success,
                error_type=(
                    None
                    if tool_result.success
                    else "ToolResultFailure"
                ),
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialize_for_llm(
                        tool_result.model_dump()
                    ),
                }
            )

    log_agent_event(
        "legacy_agent_iteration_limit_reached",
        attempt=MAX_ITERATIONS,
        success=False,
        error_type="IterationLimitExceeded",
    )

    raise RuntimeError(
        f"Agent exceeded the maximum of {MAX_ITERATIONS} iterations."
    )