import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.nodes import (
    transaction_agent_node,
)


state = {
    "messages": [
        HumanMessage(
            content=(
                "Investigate transaction TXN1001 "
                "and provide its transaction details."
            )
        )
    ],
    "route": "transaction",
    "approval_decision": None,
}


result = transaction_agent_node(state)

print("===== TRANSACTION AGENT =====")

for message in result["messages"]:
    print(message)