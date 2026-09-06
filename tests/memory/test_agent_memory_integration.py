from uuid import uuid4

from langchain_core.messages import AIMessage

from banking_investigator.agents import nodes
from banking_investigator.agents.state import AgentState
from banking_investigator.memory.postgres_store import PostgresMemoryStore
from banking_investigator.models.tool_result import ToolResult


def test_account_tool_node_writes_verified_account_memory(
    monkeypatch,
) -> None:
    customer_id = f"CUST{uuid4().hex[:8].upper()}"
    account_id = f"ACC{uuid4().hex[:8].upper()}"

    def fake_execute_tool(
        tool_name: str,
        arguments: dict,
    ) -> ToolResult:
        assert tool_name == "get_account"
        assert arguments["account_id"] == account_id

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

    state: AgentState = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_account",
                        "args": {
                            "account_id": account_id,
                        },
                        "id": "test-tool-call",
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
    }

    store = PostgresMemoryStore()

    try:
        result = nodes.account_tool_node(state)

        assert result["investigation_data"][0]["result"]["success"] is True

        memory = store.retrieve(
            namespace=f"customer:{customer_id}",
            key="account_profile",
        )

        assert memory is not None
        assert memory.value == {
            "account_id": account_id,
            "account_type": "checking",
            "status": "active",
        }

    finally:
        store.delete(
            namespace=f"customer:{customer_id}",
            key="account_profile",
        )
