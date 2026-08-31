import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.nodes import router_node


test_cases = [
    "Investigate transaction TXN1001 and identify its associated account.",
    "Show me the details of account ACC1001.",
    "Give me information about customer CUST1001.",
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
    }

    result = router_node(state)

    print("ROUTE:")
    print(result["route"])