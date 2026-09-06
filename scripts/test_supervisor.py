import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage
from langgraph.store.postgres import PostgresStore

from banking_investigator.agents.nodes import supervisor_node
from banking_investigator.config.settings import settings

test_cases = [
    "Investigate transaction TXN1001.",
    "Show me the details of account ACC1001.",
    "The investigation is complete. There is nothing else to check.",
]


with PostgresStore.from_conn_string(
    settings.postgres_conn_string
) as store:
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
            "investigation_data": [],
            "customer_id": None,
        }

        result = supervisor_node(state, store=store)

        print("NEXT AGENT:")
        print(result["next_agent"])
