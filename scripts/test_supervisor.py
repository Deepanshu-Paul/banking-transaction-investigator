import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.nodes import supervisor_node


test_cases = [
    "Investigate transaction TXN1001.",
    "Show me the details of account ACC1001.",
    "The investigation is complete. There is nothing else to check.",
]


for request in test_cases:
    print()
    print("=" * 60)
    print("REQUEST:")
    print(request)

    state = {
        "messages": [
            HumanMessage(content=request)
        ],
        "route": None,
        "approval_decision": None,
        "next_agent": None,
    }

    result = supervisor_node(state)

    print("NEXT AGENT:")
    print(result["next_agent"])