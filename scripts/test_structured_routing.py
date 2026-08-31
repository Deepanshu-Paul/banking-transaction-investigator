import sys

sys.path.insert(0, "src")

from banking_investigator.agents.routing import RouteDecision
from banking_investigator.llm.factory import get_llm_client
from langchain_core.messages import HumanMessage


llm_client = get_llm_client()


test_messages = [
    HumanMessage(
        content=(
            "Classify this request. "
            "The user wants to investigate transaction TXN1001 "
            "and identify its associated account."
        )
    )
]


decision = llm_client.invoke_structured(
    messages=test_messages,
    output_schema=RouteDecision,
)


print("===== STRUCTURED ROUTING TEST =====")
print("Provider:", type(llm_client).__name__)
print("Decision:", decision)
print("Route:", decision.route)