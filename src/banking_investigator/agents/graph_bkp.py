import psycopg

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from banking_investigator.agents.nodes import (
    human_approval_node,
    llm_node,
    router_node,
    tool_node,
)
from banking_investigator.agents.state import AgentState
from banking_investigator.config.settings import settings


def route_after_router(state: AgentState) -> str:
    route = state["route"]

    if route == "transaction":
        return "transaction"

    if route == "account":
        return "account"

    if route == "customer":
        return "customer"

    raise ValueError(
        f"Unexpected route: {route}"
    )


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


def approved_node(state: AgentState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content="Human approval received. Investigation approved."
            )
        ]
    }


def rejected_node(state: AgentState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content="Human rejection received. Investigation rejected."
            )
        ]
    }


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

# Nodes
builder.add_node("router", router_node)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_node("approval", human_approval_node)
builder.add_node("approved", approved_node)
builder.add_node("rejected", rejected_node)

# Start → Router
builder.add_edge(START, "router")

# Router → workflow
builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "transaction": "llm",
        "account": "llm",
        "customer": "llm",
    },
)

# LLM → Tools / Approval
builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: "approval",
    },
)

# Tools → LLM
builder.add_edge("tools", "llm")

# Approval → Approved / Rejected
builder.add_conditional_edges(
    "approval",
    lambda state: state["approval_decision"],
    {
        "approve": "approved",
        "reject": "rejected",
    },
)

# Final nodes
builder.add_edge("approved", END)
builder.add_edge("rejected", END)


connection = psycopg.connect(
    settings.postgres_conn_string,
    autocommit=True,
)

checkpointer = PostgresSaver(connection)

graph = builder.compile(
    checkpointer=checkpointer,
)