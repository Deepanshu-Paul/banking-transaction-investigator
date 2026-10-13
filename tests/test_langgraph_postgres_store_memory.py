import json
import logging
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore
from banking_investigator.utils.observability import LOGGER_NAME
from banking_investigator.agents import nodes
from banking_investigator.agents.graph import (
    checkpointer,
    embeddings,
    graph,
    store,
)
from banking_investigator.config.settings import settings
from banking_investigator.models.tool_result import ToolResult


@pytest.fixture(scope="module", autouse=True)
def setup_langgraph_postgres_tables() -> None:
    """Initialize LangGraph-managed PostgreSQL tables."""
    checkpointer.setup()
    store.setup()


@pytest.fixture
def semantic_store():
    with PostgresStore.from_conn_string(
        settings.postgres_conn_string,
        index={
            "dims": 1536,
            "embed": embeddings,
            "fields": ["text"],
        },
    ) as memory_store:
        memory_store.setup()
        yield memory_store


def test_postgres_store_writes_and_reads_customer_memory(
    semantic_store,
) -> None:
    namespace = (
        "customer",
        f"CUST{uuid4().hex[:8].upper()}",
    )

    profile = {
        "text": (
            "Customer CUST1001 has a SAVINGS account "
            "ACC1001 which is ACTIVE."
        ),
        "account_id": "ACC1001",
        "account_type": "SAVINGS",
        "status": "ACTIVE",
    }

    try:
        semantic_store.put(
            namespace,
            "account_profile",
            profile,
        )

        item = semantic_store.get(
            namespace,
            "account_profile",
        )

        assert item is not None
        assert item.namespace == namespace
        assert item.key == "account_profile"
        assert item.value == profile
    finally:
        semantic_store.delete(
            namespace,
            "account_profile",
        )


def test_semantic_search_finds_account_profile(
    semantic_store,
) -> None:
    customer_id = f"CUST{uuid4().hex[:8].upper()}"
    namespace = ("customer", customer_id)

    try:
        semantic_store.put(
            namespace,
            "account_profile",
            {
                "text": (
                    f"Customer {customer_id} has a SAVINGS account "
                    "ACC1001 which is ACTIVE."
                ),
                "account_id": "ACC1001",
                "account_type": "SAVINGS",
                "status": "ACTIVE",
                "memory_type": "customer_profile",
            },
        )

        memories = semantic_store.search(
            namespace,
            query="Which of this customer's bank accounts can still be used?",
            limit=5,
        )

        assert [memory.key for memory in memories] == [
            "account_profile"
        ]
        assert memories[0].value["account_id"] == "ACC1001"
    finally:
        semantic_store.delete(namespace, "account_profile")


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
            "text": (
                f"Customer {customer_id} has a SAVINGS account "
                f"{account_id} which is ACTIVE."
            ),
            "account_id": account_id,
            "account_type": "SAVINGS",
            "status": "ACTIVE",
            "memory_type": "customer_profile",
        }

    finally:
        store.delete(
            ("customer", customer_id),
            "account_profile",
        )


def test_customer_memory_persists_across_graph_runs(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(
    logging.INFO,
    logger=LOGGER_NAME,
    )
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
                # Verify trace information on the real compiled graph run.
        assert first_result["trace_id"]
        assert first_result["run_status"] == "completed"

        payloads = [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == LOGGER_NAME
        ]

        trace_id = first_result["trace_id"]

        trace_events = [
            payload
            for payload in payloads
            if payload.get("trace_id") == trace_id
        ]

        assert any(
            payload["event"] == "agent_trace_started"
            for payload in trace_events
        )

        assert any(
            payload["event"] == "agent_trace_finished"
            and payload["success"] is True
            for payload in trace_events
        )

        assert any(
            payload["event"] == "node_completed"
            and payload.get("node_name") == "supervisor"
            for payload in trace_events
        )

        assert any(
            payload["event"] == "tool_call_completed"
            for payload in trace_events
        )
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


def test_search_filters_by_memory_type(
    semantic_store,
) -> None:
    customer_id = f"CUST{uuid4().hex[:8].upper()}"
    namespace = ("customer", customer_id)

    try:
        semantic_store.put(
            namespace,
            "account_profile",
            {
                "text": "Customer has an active savings account.",
                "account_id": "ACC1001",
                "memory_type": "customer_profile",
            },
        )

        semantic_store.put(
            namespace,
            "customer_preference",
            {
                "text": "Customer prefers email communication.",
                "memory_type": "customer_preference",
            },
        )

        memories = semantic_store.search(
            namespace,
            query="customer information",
            filter={"memory_type": "customer_profile"},
            limit=5,
        )

        assert [memory.key for memory in memories] == [
            "account_profile"
        ]

    finally:
        semantic_store.delete(
            namespace,
            "account_profile",
        )
        semantic_store.delete(
            namespace,
            "customer_preference",
        )

def test_retrieve_customer_memory_filters_by_type() -> None:
    class FakeMemory:
        namespace = ("customer", "CUST1001")
        key = "account_profile"
        value = {
            "memory_type": "customer_profile",
            "account_id": "ACC1001",
        }
        score = 0.92

    class WeakMemory:
        namespace = ("customer", "CUST1001")
        key = "old_profile"
        value = {
            "memory_type": "customer_profile",
            "account_id": "OLD1001",
        }
        score = 0.10

    class FakeStore:
        def search(self, namespace, *, query, filter, limit):
            assert namespace == ("customer", "CUST1001")
            assert query == "show account"
            assert filter == {"memory_type": "customer_profile"}
            assert limit == 5

            return [FakeMemory(), WeakMemory()]

    result = nodes.retrieve_customer_memory(
        runtime=Runtime(store=FakeStore()),
        customer_id="CUST1001",
        query="show account",
        memory_type="customer_profile",
    )
    assert result == [
        {
            "namespace": ("customer", "CUST1001"),
            "key": "account_profile",
            "value": {
                "memory_type": "customer_profile",
                "account_id": "ACC1001",
            },
            "score": 0.92,
        },
    ]

def test_format_customer_memory() -> None:
    memory_items = [
        {
            "namespace": ("customer", "CUST1001"),
            "key": "account_profile",
            "value": {
                "account_id": "ACC1001",
                "account_type": "SAVINGS",
                "status": "ACTIVE",
            },
            "score": 0.92,
        }
    ]

    result = nodes.format_customer_memory(memory_items)

    assert "Customer Memory:" in result
    assert "customer_id=CUST1001" in result
    assert "account_id=ACC1001" in result
    assert "account_type=SAVINGS" in result
    assert "status=ACTIVE" in result
    assert "relevance=0.92" in result

def test_format_customer_memory_when_empty() -> None:
    result = nodes.format_customer_memory([])

    assert result == "No relevant customer memory found."

def test_blocked_graph_run_preserves_trace_id(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger=LOGGER_NAME,
    )

    def fake_invoke(
        messages,
        tools=None,
    ):
        # Deliberately request an account tool from the
        # transaction worker, which is not authorized to use it.
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_account",
                    "args": {
                        "account_id": "ACC-TEST",
                    },
                    "id": f"unauthorized-call-{uuid4()}",
                    "type": "tool_call",
                }
            ],
        )

    def fake_invoke_structured(
        messages,
        output_schema,
    ):
        return type(
            "Decision",
            (),
            {"next_agent": "transaction"},
        )()

    def fail_if_tool_executes(
        tool_name,
        arguments,
    ):
        raise AssertionError(
            f"Unauthorized tool executed: {tool_name}"
        )

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

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        fail_if_tool_executes,
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Retrieve account details"
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
                "thread_id": str(uuid4()),
            }
        },
    )

    assert result["trace_id"]
    assert result["run_status"] == "blocked"

    payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == LOGGER_NAME
    ]

    trace_id = result["trace_id"]

    trace_events = [
        payload
        for payload in payloads
        if payload.get("trace_id") == trace_id
    ]

    assert any(
        payload["event"] == "agent_trace_started"
        for payload in trace_events
    )

    finished_events = [
        payload
        for payload in trace_events
        if payload["event"] == "agent_trace_finished"
    ]

    assert len(finished_events) == 1
    assert finished_events[0]["status"] == "blocked"
    assert finished_events[0]["success"] is False

    assert any(
        payload["event"] == "tool_batch_rejected"
        for payload in trace_events
    )

    assert any(
        payload["event"] == "node_completed"
        and payload.get("node_name") == "security_abort"
        for payload in trace_events
    )

# def test_observe_memory_similarity_scores(
#     semantic_store,
# ) -> None:
#     customer_id = f"CUST{uuid4().hex[:8].upper()}"
#     namespace = ("customer", customer_id)

#     try:
#         semantic_store.put(
#             namespace,
#             "account_profile",
#             {
#                 "text": (
#                     f"Customer {customer_id} has a SAVINGS account "
#                     "ACC1001 which is ACTIVE."
#                 ),
#                 "account_id": "ACC1001",
#                 "account_type": "SAVINGS",
#                 "status": "ACTIVE",
#                 "memory_type": "customer_profile",
#             },
#         )

#         relevant = semantic_store.search(
#             namespace,
#             query="What is my savings account status?",
#             filter={"memory_type": "customer_profile"},
#             limit=1,
#         )

#         unrelated = semantic_store.search(
#             namespace,
#             query="What is the weather in London?",
#             filter={"memory_type": "customer_profile"},
#             limit=1,
#         )

#         related = semantic_store.search(
#                     namespace,
#                     query="Tell me about my account",
#                     filter={"memory_type": "customer_profile"},
#                     limit=1,
#                 )

#         print("\nRelevant score:", relevant[0].score)
#         print("Unrelated score:", unrelated[0].score)
#         print("Related score:", related[0].score)

#     finally:
#         semantic_store.delete(
#             namespace,
#             "account_profile",
#         )
        
