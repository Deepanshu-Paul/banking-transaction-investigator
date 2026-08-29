import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.graph import graph


config = {
    "configurable": {
        "thread_id": "checkpoint-demo-001",
    }
}


result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What was the account ID associated with that transaction?"
            )
        ]
    },
    config=config,
)

print("\n===== FIRST PROCESS =====")
print("Message count:", len(result["messages"]))
print("Last message:")
print(result["messages"][-1].content)