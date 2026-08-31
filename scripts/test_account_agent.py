import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.nodes import account_agent_node


state = {
    "messages": [
        HumanMessage(
            content=(
                "Show me the details of account ACC1001."
            )
        )
    ],
    "route": "account",
    "approval_decision": None,
}


result = account_agent_node(state)

print("===== ACCOUNT AGENT =====")

for message in result["messages"]:
    print(message)