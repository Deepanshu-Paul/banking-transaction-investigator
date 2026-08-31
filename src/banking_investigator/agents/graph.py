import psycopg

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

from banking_investigator.agents.nodes import (
    account_agent_node,
    supervisor_node,
    transaction_agent_node,
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
        "finish": END,
    },
)


# -------------------------
# Workers → END (temporary)
# -------------------------

builder.add_edge(
    "transaction_agent",
    END,
)

builder.add_edge(
    "account_agent",
    END,
)


# -------------------------
# PostgreSQL checkpointing
# -------------------------

connection = psycopg.connect(
    settings.postgres_conn_string,
    autocommit=True,
)

checkpointer = PostgresSaver(connection)


graph = builder.compile(
    checkpointer=checkpointer,
)