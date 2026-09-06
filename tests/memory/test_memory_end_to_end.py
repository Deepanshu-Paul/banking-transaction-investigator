from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from banking_investigator.agents import nodes
from banking_investigator.agents.state import AgentState
from banking_investigator.memory.postgres_store import PostgresMemoryStore
from banking_investigator.models.tool_result import ToolResult


def test_memory_survives_from_account_lookup_to_later_supervisor_run(
    monkeypatch,
) -> None:
    customer_id = f"CUST{uuid4().hex[:8].upper()}"
    account_id = f"ACC{uuid4().hex[:8].upper()}"

    def fake_execute_tool(
        tool_name: str,
        arguments: dict,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "account_id": account_id,
                "customer_id": customer_id,
                "account_type": "checking",
                "status": "active",
            },
        )

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        fake_execute_tool,
    )

    first_run_state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_account",
                        "args": {
                            "account_id": account_id,
                        },
                        "id": "first-run-tool-call",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "route": None,
        "approval_decision": None,
        "next_agent": None,
        "investigation_data": [],
        "memory": [],
        "customer_id": None,
    }

    store = PostgresMemoryStore()

    try:
        account_result = nodes.account_tool_node(
            first_run_state
        )

        assert account_result["customer_id"] == customer_id

        class FakeDecision:
            next_agent = "finish"

        def fake_invoke_structured(
            messages,
            output_schema,
        ):
            return FakeDecision()

        monkeypatch.setattr(
            nodes.llm_client,
            "invoke_structured",
            fake_invoke_structured,
        )

        second_run_state: AgentState = {
            "messages": [
                HumanMessage(
                    content="Investigate account"
                )
            ],
            "route": None,
            "approval_decision": None,
            "next_agent": None,
            "investigation_data": [],
            "memory": [],
            "customer_id": customer_id,
        }

        supervisor_result = nodes.supervisor_node(
            second_run_state
        )

        assert supervisor_result["memory"] == [
            {
                "namespace": f"customer:{customer_id}",
                "key": "account_profile",
                "value": {
                    "account_id": account_id,
                    "account_type": "checking",
                    "status": "active",
                },
            }
        ]

    finally:
        store.delete(
            namespace=f"customer:{customer_id}",
            key="account_profile",
        )
