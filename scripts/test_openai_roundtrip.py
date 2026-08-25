import sys

sys.path.insert(0, "src")

from banking_investigator.llm.factory import get_llm_client
from langchain_core.messages import HumanMessage, ToolMessage
from banking_investigator.tools.schema import GET_TRANSACTION_TOOL


client = get_llm_client()

first = client.invoke(
    [
        HumanMessage(
            content="Get the details for transaction TXN1001."
        )
    ],
    tools=[GET_TRANSACTION_TOOL],
)

print("FIRST:")
print(first)
print()

tool_call = first.tool_calls[0]

tool_result = (
    '{"success": true, '
    '"data": {'
    '"transaction_id": "TXN1001", '
    '"account_id": "ACC1001", '
    '"amount": "2500.00", '
    '"currency": "INR", '
    '"merchant": "Amazon", '
    '"transaction_type": "CARD_PAYMENT", '
    '"status": "APPROVED"'
    '}, '
    '"error": null}'
)

second = client.invoke(
    [
        HumanMessage(
            content="Get the details for transaction TXN1001."
        ),
        first,
        ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"],
        ),
    ],
    tools=[GET_TRANSACTION_TOOL],
)

print("SECOND:")
print(second)
print()

print("CONTENT:")
print(second.content)

print()

print("TOOL CALLS:")
print(second.tool_calls)