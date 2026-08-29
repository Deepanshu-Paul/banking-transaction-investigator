from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import interrupt

from banking_investigator.agents.state import AgentState
from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.llm.factory import get_llm_client
from banking_investigator.tools.schema import (
    GET_ACCOUNT_TOOL,
    GET_TRANSACTION_TOOL,
)
from banking_investigator.utils.serialization import serialize_for_llm


TOOLS = [
    GET_TRANSACTION_TOOL,
    GET_ACCOUNT_TOOL,
]


llm_client = get_llm_client()


def llm_node(state: AgentState) -> dict:
    response = llm_client.invoke(
        state["messages"],
        tools=TOOLS,
    )

    return {
        "messages": [response],
    }


def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]

        tool_result = execute_tool(
            tool_name,
            arguments,
        )

        tool_messages.append(
            ToolMessage(
                content=serialize_for_llm(
                    tool_result.model_dump()
                ),
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "messages": tool_messages,
    }


def human_approval_node(state: AgentState) -> dict:
    decision = interrupt(
        {
            "type": "human_approval",
            "message": (
                "Please review the investigation "
                "and approve or reject it."
            ),
            "options": [
                "approve",
                "reject",
            ],
        }
    )

    if decision not in {"approve", "reject"}:
        raise ValueError(
            "Human decision must be 'approve' or 'reject'."
        )

    return {
        "approval_decision": decision,
    }


def approved_node(state: AgentState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Human approval received. "
                    "Investigation approved."
                )
            )
        ]
    }


def rejected_node(state: AgentState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Human rejection received. "
                    "Investigation rejected."
                )
            )
        ]
    }