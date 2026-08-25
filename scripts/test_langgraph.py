import sys

sys.path.insert(0, "src")

from banking_investigator.agents.graph import graph


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
    }
)

print("\nFINAL MESSAGES:\n")

for message in result["messages"]:
    print(message)