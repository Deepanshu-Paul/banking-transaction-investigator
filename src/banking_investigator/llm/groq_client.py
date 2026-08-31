import json
from typing import Any, TypeVar

from groq import Groq
from langchain_core.messages import AIMessage

from banking_investigator.config.settings import settings
from banking_investigator.llm.base import LLMClient


T = TypeVar("T")


class GroqLLMClient(LLMClient):

    def __init__(self) -> None:
        self.client = Groq(
            api_key=settings.llm_api_key
        )

    def _to_groq_message(self, message) -> dict:
        if message.type == "human":
            return {
                "role": "user",
                "content": message.content,
            }

        if message.type == "system":
            return {
                "role": "system",
                "content": message.content,
            }

        if message.type == "ai":
            result = {
                "role": "assistant",
                "content": message.content or "",
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(
                                tool_call["args"]
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            return result

        if message.type == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        raise ValueError(
            f"Unsupported message type: {message.type}"
        )

    def invoke(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
    ) -> AIMessage:

        groq_messages = [
            self._to_groq_message(message)
            for message in messages
        ]

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=groq_messages,
            tools=tools,
            tool_choice="auto",
        )

        groq_message = response.choices[0].message

        return AIMessage(
            content=groq_message.content or "",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "args": json.loads(
                        tool_call.function.arguments
                    ),
                }
                for tool_call in (groq_message.tool_calls or [])
            ],
        )

    def invoke_structured(
        self,
        messages: list[Any],
        output_schema: type[T],
    ) -> T:

        groq_messages = [
            self._to_groq_message(message)
            for message in messages
        ]

        schema = output_schema.model_json_schema()

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=groq_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Groq returned an empty structured response."
            )

        data = json.loads(content)

        return output_schema.model_validate(data)