import sys

sys.path.insert(0, "src")

from banking_investigator.config.settings import settings
from banking_investigator.llm_client import client
from banking_investigator.tools.schema import (
    GET_ACCOUNT_TOOL,
    GET_TRANSACTION_TOOL,
)


response = client.chat.completions.create(
    model=settings.llm_model,
    messages=[
        {
            "role": "user",
            "content": "I need information about account ACC1001.",
        }
    ],
    tools=[
        GET_TRANSACTION_TOOL,
        GET_ACCOUNT_TOOL,
    ],
)

message = response.choices[0].message

print("CONTENT:")
print(message.content)

print("\nTOOL CALLS:")
print(message.tool_calls)