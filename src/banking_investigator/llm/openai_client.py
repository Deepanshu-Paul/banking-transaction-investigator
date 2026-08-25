import json
from typing import Any

from langchain_core.messages import AIMessage
from openai import OpenAI

from banking_investigator.config.settings import settings
from banking_investigator.llm.base import LLMClient


class OpenAILLMClient(LLMClient):

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.llm_api_key
        )

    def _to_openai_input(self, message) -> list[dict]:
        if message.type == "human":
            return [
                {
                    "role": "user",
                    "content": message.content,
                }
            ]

        if message.type == "system":
            return [
                {
                    "role": "system",
                    "content": message.content,
                }
            ]

        if message.type == "ai":
            # Responses API function-call items must be preserved
            # for the following function_call_output.
            openai_output = message.additional_kwargs.get(
                "openai_response_output"
            )

            if openai_output:
                return openai_output

            return [
                {
                    "role": "assistant",
                    "content": message.content or "",
                }
            ]

        if message.type == "tool":
            return [
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            ]

        raise ValueError(
            f"Unsupported message type: {message.type}"
        )

    def _to_openai_tools(
        self,
        tools: list[dict] | None,
    ) -> list[dict] | None:

        if not tools:
            return None

        converted_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]

                converted_tools.append(
                    {
                        "type": "function",
                        "name": function["name"],
                        "description": function.get(
                            "description",
                            "",
                        ),
                        "parameters": function.get(
                            "parameters",
                            {},
                        ),
                    }
                )
            else:
                converted_tools.append(tool)

        return converted_tools

    def invoke(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
    ) -> AIMessage:

        openai_input = []

        for message in messages:
            openai_input.extend(
                self._to_openai_input(message)
            )

        response = self.client.responses.create(
            model=settings.llm_model,
            input=openai_input,
            tools=self._to_openai_tools(tools),
        )

        content = response.output_text or ""

        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    {
                        "name": item.name,
                        "args": json.loads(item.arguments),
                        "id": item.call_id,
                        "type": "tool_call",
                    }
                )

        openai_output = []

        for item in response.output:
            if item.type == "function_call":
                openai_output.append(
                    {
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                )

            elif item.type == "message":
                openai_output.append(
                    {
                        "type": "message",
                        "role": item.role,
                        "content": [
                            content.model_dump()
                            for content in item.content
                        ],
                    }
                )

        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs={
                "openai_response_output": openai_output
            },
        )