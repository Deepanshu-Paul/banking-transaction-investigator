from langgraph.graph import END, START, StateGraph

from banking_investigator.agents.nodes import llm_node, tool_node
from banking_investigator.agents.state import AgentState


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


builder = StateGraph(AgentState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "llm")

graph = builder.compile()