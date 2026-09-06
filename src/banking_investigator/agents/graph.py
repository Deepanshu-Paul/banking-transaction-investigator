import psycopg

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres import PostgresStore

from banking_investigator.agents.nodes import (
    account_agent_node,
    account_tool_node,
    final_response_node,
    supervisor_node,
    transaction_agent_node,
    transaction_tool_node,
)
from banking_investigator.agents.state import AgentState
from banking_investigator.config.settings import settings


def route_after_supervisor(state: AgentState) -> str:
    next_agent = state["next_agent"]

    if next_agent == "transaction":
        return "transaction"

    if next_agent == "account":
        return "account"

    if next_agent == "finish":
        return "finish"

    raise ValueError(
        f"Unexpected supervisor decision: {next_agent}"
    )


def should_continue_transaction(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "supervisor"


def should_continue_account(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "supervisor"


builder = StateGraph(AgentState)

# -------------------------
# Nodes
# -------------------------

builder.add_node(
    "supervisor",
    supervisor_node,
)

builder.add_node(
    "transaction_agent",
    transaction_agent_node,
)

builder.add_node(
    "account_agent",
    account_agent_node,
)

builder.add_node(
    "transaction_tools",
    transaction_tool_node,
)

builder.add_node(
    "account_tools",
    account_tool_node,
)

builder.add_node(
    "final_response",
    final_response_node,
)

# -------------------------
# START → SUPERVISOR
# -------------------------

builder.add_edge(
    START,
    "supervisor",
)

# -------------------------
# SUPERVISOR → WORKER
# -------------------------

builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "transaction": "transaction_agent",
        "account": "account_agent",
        "finish": "final_response",
    },
)

builder.add_edge(
    "final_response",
    END,
)

# -------------------------
# WORKER → TOOL / SUPERVISOR
# -------------------------

builder.add_conditional_edges(
    "transaction_agent",
    should_continue_transaction,
    {
        "tools": "transaction_tools",
        "supervisor": "supervisor",
    },
)

builder.add_conditional_edges(
    "account_agent",
    should_continue_account,
    {
        "tools": "account_tools",
        "supervisor": "supervisor",
    },
)

# -------------------------
# TOOL → WORKER
# -------------------------

builder.add_edge(
    "transaction_tools",
    "transaction_agent",
)

builder.add_edge(
    "account_tools",
    "account_agent",
)

# -------------------------
# PostgreSQL persistence
# -------------------------

connection = psycopg.connect(
    settings.postgres_conn_string,
    autocommit=True,
)

checkpointer = PostgresSaver(connection)
store = PostgresStore(connection)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)