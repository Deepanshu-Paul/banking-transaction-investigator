import json
from banking_investigator.models.tool_result import ToolResult
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from banking_investigator.agents import nodes


def make_tool_call(name, arguments, call_id):
    return {
        "name": name,
        "args": arguments,
        "id": call_id,
        "type": "tool_call",
    }


def test_rejects_entire_batch_before_executing_any_tool(
    monkeypatch,
):
    executed_tools = []

    def fake_execute_tool(tool_name, arguments):
        executed_tools.append(tool_name)
        raise AssertionError(
            "No tool should execute when the batch is unauthorized."
        )

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        fake_execute_tool,
    )

    message = AIMessage(
        content="",
        tool_calls=[
            make_tool_call(
                "get_transaction",
                {"transaction_id": "TXN-123"},
                "call-1",
            ),
            make_tool_call(
                "get_account",
                {"account_id": "ACC-123"},
                "call-2",
            ),
        ],
    )

    result = nodes._execute_scoped_tools(
        {"messages": [message]},
        {"get_transaction"},
    )

    assert executed_tools == []
    assert result["investigation_data"] == []

    tool_messages = result["messages"]

    assert len(tool_messages) == 2
    assert [
        message.tool_call_id for message in tool_messages
    ] == ["call-1", "call-2"]

    for tool_message in tool_messages:
        payload = json.loads(tool_message.content)

        assert payload["success"] is False
        assert "No tools in this batch were executed" in (
            payload["error"]
        )


def test_first_unauthorized_attempt_can_be_rejected_for_recovery():
    state = {
        "messages": [
            HumanMessage(content="Look up my transaction."),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "get_account",
                        {"account_id": "ACC-123"},
                        "call-1",
                    )
                ],
            ),
        ]
    }

    assert nodes.should_continue_worker(
        state,
        {"get_transaction"},
    ) == "tools"


def test_repeated_unauthorized_attempt_aborts_same_turn():
    state = {
        "messages": [
            HumanMessage(content="Look up my transaction."),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "get_account",
                        {"account_id": "ACC-123"},
                        "call-1",
                    )
                ],
            ),
            ToolMessage(
                content=(
                    '{"success": false, '
                    '"error": "Tool call batch rejected."}'
                ),
                tool_call_id="call-1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "get_account",
                        {"account_id": "ACC-456"},
                        "call-2",
                    )
                ],
            ),
        ]
    }

    assert nodes.should_continue_worker(
        state,
        {"get_transaction"},
    ) == "abort"


def test_new_user_turn_resets_rejection_guard():
    state = {
        "messages": [
            HumanMessage(content="Earlier request."),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "get_account",
                        {"account_id": "ACC-123"},
                        "call-1",
                    )
                ],
            ),
            ToolMessage(
                content=(
                    '{"success": false, '
                    '"error": "Tool call batch rejected."}'
                ),
                tool_call_id="call-1",
            ),
            HumanMessage(content="A new request."),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "get_account",
                        {"account_id": "ACC-456"},
                        "call-2",
                    )
                ],
            ),
        ]
    }

    assert nodes.should_continue_worker(
        state,
        {"get_transaction"},
    ) == "tools"


def test_security_abort_node_closes_tool_calls_and_stops():
    last_message = AIMessage(
        content="",
        tool_calls=[
            make_tool_call(
                "get_account",
                {"account_id": "ACC-123"},
                "call-1",
            ),
            make_tool_call(
                "get_transaction",
                {"transaction_id": "TXN-456"},
                "call-2",
            ),
        ],
    )

    result = nodes.security_abort_node(
        {"messages": [last_message]},
    )

    assert len(result["messages"]) == 3

    tool_messages = result["messages"][:2]
    final_message = result["messages"][2]

    assert [message.tool_call_id for message in tool_messages] == [
        "call-1",
        "call-2",
    ]
    assert all(
        json.loads(message.content)["success"] is False
        for message in tool_messages
    )

    assert isinstance(final_message, AIMessage)
    assert "repeatedly requested tools" in final_message.content

def test_authorized_tool_executes_successfully(monkeypatch):
    executed_tools = []

    def fake_execute_tool(tool_name, arguments):
        executed_tools.append((tool_name, arguments))
        return ToolResult(
            success=True,
            data={"transaction_id": "TXN-123"},
        )

    monkeypatch.setattr(
        nodes,
        "execute_tool",
        fake_execute_tool,
    )

    message = AIMessage(
        content="",
        tool_calls=[
            make_tool_call(
                "get_transaction",
                {"transaction_id": "TXN-123"},
                "call-allowed",
            )
        ],
    )

    result = nodes._execute_scoped_tools(
        {"messages": [message]},
        {"get_transaction"},
    )

    assert executed_tools == [
        (
            "get_transaction",
            {"transaction_id": "TXN-123"},
        )
    ]
    assert result["investigation_data"][0]["result"]["success"]
    assert result["messages"][0].tool_call_id == "call-allowed"