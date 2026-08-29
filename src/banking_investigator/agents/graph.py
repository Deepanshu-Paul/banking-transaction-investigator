import psycopg

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from banking_investigator.agents.nodes import (
    approved_node,
    human_approval_node,
    llm_node,
    rejected_node,
    tool_node,
)
from banking_investigator.agents.state import AgentState
from banking_investigator.config.settings import settings


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


def route_after_approval(state: AgentState) -> str:
    decision = state["approval_decision"]

    if decision == "approve":
        return "approved"

    if decision == "reject":
        return "rejected"

    raise ValueError(
        f"Unexpected approval decision: {decision}"
    )


builder = StateGraph(AgentState)


# -------------------------
# Nodes
# -------------------------

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_node("approval", human_approval_node)
builder.add_node("approved", approved_node)
builder.add_node("rejected", rejected_node)


# -------------------------
# Entry point
# -------------------------

builder.add_edge(
    START,
    "llm",
)


# -------------------------
# LLM → Tools / Approval
# -------------------------

builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: "approval",
    },
)


# -------------------------
# Tools → LLM
# -------------------------

builder.add_edge(
    "tools",
    "llm",
)


# -------------------------
# Approval → Approved / Rejected
# -------------------------

builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "approved": "approved",
        "rejected": "rejected",
    },
)


# -------------------------
# Terminal paths
# -------------------------

builder.add_edge(
    "approved",
    END,
)

builder.add_edge(
    "rejected",
    END,
)


# -------------------------
# PostgreSQL Checkpointer
# -------------------------

connection = psycopg.connect(
    settings.postgres_conn_string,
    autocommit=True,
)

checkpointer = PostgresSaver(
    connection,
)


# -------------------------
# Compile graph
# -------------------------

graph = builder.compile(
    checkpointer=checkpointer,
)