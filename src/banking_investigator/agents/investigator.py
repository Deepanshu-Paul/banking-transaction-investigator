import json

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.config.settings import settings
from banking_investigator.llm_client import client
from banking_investigator.utils.serialization import serialize_for_llm
from banking_investigator.config.logging import get_logger
from banking_investigator.tools.schema import (
    GET_ACCOUNT_TOOL,
    GET_TRANSACTION_TOOL,
)

logger = get_logger(__name__)

TOOLS = [
    GET_TRANSACTION_TOOL,
    GET_ACCOUNT_TOOL,
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
        logger.info("Agent iteration=%s", iteration)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            logger.info("Agent completed")
            return message.content or ""

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            logger.info(
                "Tool requested name=%s arguments=%s",
                tool_name,
                arguments,
            )

            tool_result = execute_tool(tool_name, arguments)

            logger.info(
                "Tool completed name=%s success=%s",
                tool_name,
                tool_result.success,
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
    raise RuntimeError(
        f"Agent exceeded the maximum of {MAX_ITERATIONS} iterations."
    )