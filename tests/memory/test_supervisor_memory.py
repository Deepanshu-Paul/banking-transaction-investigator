from uuid import uuid4

from langchain_core.messages import HumanMessage

from banking_investigator.agents import nodes
from banking_investigator.memory.policy import MemoryType
from banking_investigator.memory.service import MemoryService


def test_supervisor_retrieves_persisted_customer_memory(
    monkeypatch,
) -> None:
    customer_id = f"CUST{uuid4().hex[:8].upper()}"

    memory_service = MemoryService()

    memory_service.remember(
        namespace=f"customer:{customer_id}",
        key="account_profile",
        value={
            "account_id": "ACC1001",
            "account_type": "checking",
            "status": "active",
        },
        memory_type=MemoryType.CUSTOMER_PROFILE,
    )

    class FakeDecision:
        next_agent = "finish"

    captured_messages = []

    def fake_invoke_structured(
        messages,
        output_schema,
    ):
        captured_messages.extend(messages)
        return FakeDecision()

    monkeypatch.setattr(
        nodes.llm_client,
        "invoke_structured",
        fake_invoke_structured,
    )

    state = {
        "messages": [
            HumanMessage(
                content="Investigate account ACC1001"
            )
        ],
        "route": None,
        "approval_decision": None,
        "next_agent": None,
        "investigation_data": [],
        "memory": [],
        "customer_id": customer_id,
    }

    try:
        result = nodes.supervisor_node(state)

        assert result["next_agent"] == "finish"

        assert result["memory"] == [
            {
                "namespace": f"customer:{customer_id}",
                "key": "account_profile",
                "value": {
                    "account_id": "ACC1001",
                    "account_type": "checking",
                    "status": "active",
                },
            }
        ]

        supervisor_prompt = captured_messages[0].content

        assert customer_id in supervisor_prompt
        assert "account_profile" in supervisor_prompt
        assert "ACC1001" in supervisor_prompt

    finally:
        memory_service.forget(
            namespace=f"customer:{customer_id}",
            key="account_profile",
        )
