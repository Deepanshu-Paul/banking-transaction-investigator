import json

from langchain_core.messages import AIMessage

from banking_investigator.agents import nodes


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
            {
                "name": "get_transaction",
                "args": {"transaction_id": "TXN-123"},
                "id": "call-1",
                "type": "tool_call",
            },
            {
                "name": "get_account",
                "args": {"account_id": "ACC-123"},
                "id": "call-2",
                "type": "tool_call",
            },
        ],
    )

    result = nodes._execute_scoped_tools(
        {"messages": [message]},
        {"get_transaction"},
    )

    # Neither tool should execute.
    assert executed_tools == []

    # The rejected batch adds no investigation data.
    assert result["investigation_data"] == []

    # Each requested tool call receives a matching rejection.
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