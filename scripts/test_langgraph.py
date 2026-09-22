import sys
from uuid import uuid4

sys.path.insert(0, "src")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from banking_investigator.agents.graph import graph

config = {
    "configurable": {
        "thread_id": str(uuid4()),
    }
}

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Investigate transaction TXN1001 and check "
                    "whether the associated account is active."
                ),
            }
        ]
    },
    config=config,
)

print("\nFINAL MESSAGES:\n")

for message in result["messages"]:
    print(message)
