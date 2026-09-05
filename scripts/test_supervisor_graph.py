import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.graph import graph


def run_test(request: str, thread_id: str) -> None:
    print("\n" + "=" * 60)
    print("REQUEST:")
    print(request)

    state = {
        "messages": [
            HumanMessage(content=request)
        ],
        "route": None,
        "approval_decision": None,
        "next_agent": None,
        "investigation_data": [],
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = graph.invoke(
        state,
        config=config,
    )

    print("\nFINAL STATE:")
    print("Next agent:", result.get("next_agent"))

    print("\nMESSAGES:")
    for message in result["messages"]:
        print(message)

run_test(
    "Investigate transaction TXN1001.",
    "final-response-test-transaction-001",
)

run_test(
    "Show me the details of account ACC1001.",
    "final-response-test-account-001",
)