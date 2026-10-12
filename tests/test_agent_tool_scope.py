import pytest
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

    state = {"messages": [message]}

    with pytest.raises(
        ValueError,
        match="unauthorized tool",
    ):
        nodes._execute_scoped_tools(
            state,
            {"get_transaction"},
        )

    assert executed_tools == []