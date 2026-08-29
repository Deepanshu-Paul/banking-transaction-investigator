import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.graph import graph


config = {
    "configurable": {
        "thread_id": "investigation-001",
    }
}


# First request
result_1 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Investigate transaction TXN1001 and check whether the associated account is active."
            )
        ]
    },
    config=config,
)

print("\n===== FIRST REQUEST =====")
print("Message count:", len(result_1["messages"]))


# Second request — SAME THREAD
result_2 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What was the account ID associated with that transaction?"
            )
        ]
    },
    config=config,
)

print("\n===== SECOND REQUEST =====")
print("Message count:", len(result_2["messages"]))
print("\nLAST MESSAGE:")
print(result_2["messages"][-1].content)