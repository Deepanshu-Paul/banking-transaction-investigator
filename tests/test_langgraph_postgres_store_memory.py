from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore

from banking_investigator.agents import nodes
from banking_investigator.agents.graph import checkpointer, graph, store
from banking_investigator.config.settings import settings
from banking_investigator.models.tool_result import ToolResult


@pytest.fixture(scope="module", autouse=True)
def setup_langgraph_postgres_tables() -> None:
    """Initialize LangGraph-managed PostgreSQL tables."""
    checkpointer.setup()
    store.setup()


def test_postgres_store_writes_and_reads_customer_memory() -> None:
    namespace = (
        "customer",
        f"CUST{uuid4().hex[:8].upper()}",
    )

    with PostgresStore.from_conn_string(
        settings.postgres_conn_string
    ) as memory_store:
        memory_store.setup()

        memory_store.put(
            namespace,
            "account_profile",
            {
                "account_id": "ACC1001",
                "account_type": "SAVINGS",
                "status": "ACTIVE",
            },
        )

        item = memory_store.get(
            namespace,
            "account_profile",
        )

        assert item is not None
        assert item.namespace == namespace
        assert item.key == "account_profile"
        assert item.value == {
            "account_id": "ACC1001",
            "account_type": "SAVINGS",
            "status": "ACTIVE",
        }

        memory_store.delete(
            namespace,
            "account_profile",
        )


def test_account_lookup_writes_customer_memory(monkeypatch) -> None:
    customer_id = (
        f"CUST{uuid4().hex[:8].upper()}"
    )
    account_id = (
        f"ACC{uuid4().hex[:8].upper()}"
    )

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        lambda _tool_name, _arguments: ToolResult(
            success=True,
            data={
                "account_id": account_id,
                "customer_id": customer_id,
                "account_type": "SAVINGS",
                "status": "ACTIVE",
            },
        ),
    )

    result = nodes.account_tool_node(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_account",
                            "args": {
                                "account_id": account_id
                            },
                            "id": "account-memory-test",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "route": None,
            "approval_decision": None,
            "next_agent": None,
            "investigation_data": [],
            "customer_id": None,
        },
        Runtime(store=store),
    )

    try:
        item = store.get(
            ("customer", customer_id),
            "account_profile",
        )

        assert result["customer_id"] == customer_id
        assert item is not None

        assert item.value == {
            "account_id": account_id,
            "account_type": "SAVINGS",
            "status": "ACTIVE",
        }

    finally:
        store.delete(
            ("customer", customer_id),
            "account_profile",
        )


def test_customer_memory_persists_across_graph_runs(
    monkeypatch,
) -> None:
    customer_id = (
        f"CUST{uuid4().hex[:8].upper()}"
    )
    account_id = (
        f"ACC{uuid4().hex[:8].upper()}"
    )

    supervisor_prompts = []

    decisions = iter(
        (
            "account",
            "finish",
            "finish",
        )
    )

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        lambda _tool_name, _arguments: ToolResult(
            success=True,
            data={
                "account_id": account_id,
                "customer_id": customer_id,
                "account_type": "SAVINGS",
                "status": "ACTIVE",
            },
        ),
    )

    def fake_invoke(
        messages,
        tools=None,
    ):
        if tools == nodes.ACCOUNT_TOOLS:
            if messages[-1].type == "tool":
                return AIMessage(
                    content="Account details retrieved."
                )

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_account",
                        "args": {
                            "account_id": account_id
                        },
                        "id": f"account-call-{uuid4()}",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content="Investigation complete."
        )

    def fake_invoke_structured(
        messages,
        output_schema,
    ):
        supervisor_prompts.append(
            messages[0].content
        )

        return type(
            "Decision",
            (),
            {
                "next_agent": next(decisions)
            },
        )()

    monkeypatch.setattr(
        nodes.llm_client,
        "invoke",
        fake_invoke,
    )

    monkeypatch.setattr(
        nodes.llm_client,
        "invoke_structured",
        fake_invoke_structured,
    )

    first_result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Show account details"
                )
            ],
            "route": None,
            "approval_decision": None,
            "next_agent": None,
            "investigation_data": [],
            "customer_id": None,
        },
        config={
            "configurable": {
                "thread_id": str(uuid4())
            }
        },
    )

    try:
        assert first_result["next_agent"] == "finish"

        assert any(
            "account_profile" in prompt
            for prompt in supervisor_prompts
        )

        graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Use saved profile"
                    )
                ],
                "route": None,
                "approval_decision": None,
                "next_agent": None,
                "investigation_data": [],
                "customer_id": customer_id,
            },
            config={
                "configurable": {
                    "thread_id": str(uuid4())
                }
            },
        )

        assert customer_id in supervisor_prompts[-1]
        assert "account_profile" in supervisor_prompts[-1]
        assert account_id in supervisor_prompts[-1]

    finally:
        store.delete(
            ("customer", customer_id),
            "account_profile",
        )