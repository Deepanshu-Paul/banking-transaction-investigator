from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import interrupt

from banking_investigator.agents.routing import (
    RouteDecision,
    SupervisorDecision,
)
from banking_investigator.memory.postgres_store import PostgresMemoryStore
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

TRANSACTION_TOOLS = [
    GET_TRANSACTION_TOOL,
]

ACCOUNT_TOOLS = [
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


def _execute_scoped_tools(
    state: AgentState,
    allowed_tools: set[str],
) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    investigation_data = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]

        if tool_name not in allowed_tools:
            raise ValueError(
                f"Tool '{tool_name}' is not allowed for this worker."
            )

        tool_result = execute_tool(
            tool_name,
            tool_call["args"],
        )

        result_data = tool_result.model_dump()

        investigation_data.append({
            "tool": tool_name,
            "result": result_data,
        })

        tool_messages.append(
            ToolMessage(
                content=serialize_for_llm(result_data),
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "messages": tool_messages,
        "investigation_data": investigation_data,
    }


def transaction_tool_node(state: AgentState) -> dict:
    return _execute_scoped_tools(
        state,
        {"get_transaction"},
    )


def account_tool_node(state: AgentState) -> dict:
    return _execute_scoped_tools(
        state,
        {"get_account"},
    )


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


def router_node(state: AgentState) -> dict:
    decision = llm_client.invoke_structured(
        messages=[
            HumanMessage(
                content=(
                    "Classify the user's request into exactly one "
                    "of these categories: transaction, account, customer."
                )
            ),
            state["messages"][0],
        ],
        output_schema=RouteDecision,
    )

    return {
        "route": decision.route,
    }


def transaction_agent_node(state: AgentState) -> dict:
    response = llm_client.invoke(
        state["messages"],
        tools=TRANSACTION_TOOLS,
    )

    return {
        "messages": [response],
    }


def account_agent_node(state: AgentState) -> dict:
    response = llm_client.invoke(
        state["messages"],
        tools=ACCOUNT_TOOLS,
    )

    return {
        "messages": [response],
    }


def final_response_node(state: AgentState) -> dict:
    messages = [
        HumanMessage(
            content=(
                "Generate the final answer to the user's request.\n\n"
                f"User request:\n{state['messages'][0].content}\n\n"
                "Use ONLY the investigation data provided below.\n"
                "Do not invent facts, risk scores, history, "
                "geolocation, device information, or other details "
                "that are not present in the investigation data.\n"
                "If the available data is insufficient to answer "
                "something, say so.\n\n"
                f"Investigation data:\n{state['investigation_data']}"
            )
        )
    ]

    response = llm_client.invoke(
        messages,
        tools=None,
    )

    return {
        "messages": [response],
    }


def supervisor_node(state: AgentState) -> dict:
    # ---------------------------------------------------------
    # Load long-term memory
    # ---------------------------------------------------------

    memory_store = PostgresMemoryStore()

    user_message = state["messages"][0].content

    memory_items = []

    customer_id = None

    if "CUST" in user_message:
        words = user_message.split()

        for word in words:
            if word.startswith("CUST"):
                customer_id = word.strip(".,!?")
                break

    if customer_id is not None:
        namespace = f"customer:{customer_id}"

        memories = memory_store.retrieve_namespace(
            namespace=namespace,
        )

        memory_items = [
            {
                "namespace": memory.namespace,
                "key": memory.key,
                "value": memory.value,
            }
            for memory in memories
        ]

    # ---------------------------------------------------------
    # Supervisor decision
    # ---------------------------------------------------------

    decision_messages = [
        HumanMessage(
            content=(
                "You are a supervisor controlling a multi-agent banking system.\n\n"
                "Your ONLY job is to choose the next worker.\n"
                "DO NOT answer the user's request.\n"
                "DO NOT summarize transaction or account data.\n"
                "DO NOT provide explanations or recommendations.\n\n"
                "Available choices:\n"
                "- transaction: send the request to the transaction worker\n"
                "- account: send the request to the account worker\n"
                "- finish: use when a worker has completed the investigation\n\n"
                "If the latest worker response contains the requested information "
                "and does not request another tool, choose 'finish'.\n"
                "If more work is required, choose the appropriate worker.\n\n"
                f"Long-term memory available for this run:\n{memory_items}\n\n"
                "Return ONLY the structured SupervisorDecision."
            )
        ),
        *state["messages"],
    ]

    decision = llm_client.invoke_structured(
        messages=decision_messages,
        output_schema=SupervisorDecision,
    )

    return {
        "next_agent": decision.next_agent,
        "memory": memory_items,
    }
