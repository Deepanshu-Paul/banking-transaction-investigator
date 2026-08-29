import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from banking_investigator.agents.graph import graph


config = {
    "configurable": {
        "thread_id": "hitl-demo-002",
    }
}


print("\n===== STARTING INVESTIGATION =====")

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Investigate transaction TXN1001 "
                    "and identify its associated account."
                )
            )
        ]
    },
    config=config,
)

print("\n===== GRAPH INTERRUPTED =====")
print(result["__interrupt__"])


print("\n===== RESUMING WITH HUMAN DECISION =====")

result = graph.invoke(
    Command(resume="reject"),
    config=config,
)

print("\n===== GRAPH RESUMED =====")

for message in result["messages"]:
    print(message)