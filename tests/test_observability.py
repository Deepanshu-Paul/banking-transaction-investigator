import json
import logging

from langchain_core.messages import AIMessage

from banking_investigator.models.tool_result import ToolResult
from banking_investigator.agents import nodes
from banking_investigator.utils.observability import (
    LOGGER_NAME,
    log_agent_event,
)


def test_log_agent_event_emits_json_and_omits_sensitive_fields(
    caplog,
):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_agent_event(
            "tool_call_completed",
            tool_name="get_transaction",
            duration_ms=12.5,
            success=True,
            transaction_id="TXN-SECRET",
            arguments={"transaction_id": "TXN-SECRET"},
        )

    records = [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME
    ]

    assert len(records) == 1

    payload = json.loads(records[0].getMessage())

    assert payload == {
        "event": "tool_call_completed",
        "tool_name": "get_transaction",
        "duration_ms": 12.5,
        "success": True,
    }

    assert "TXN-SECRET" not in records[0].getMessage()


def test_tool_execution_emits_start_and_completion_events(
    monkeypatch,
    caplog,
):
    def fake_execute_tool(tool_name, arguments):
        return ToolResult(
            success=True,
            data={"transaction_id": "TXN-SECRET"},
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
                "args": {"transaction_id": "TXN-SECRET"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = nodes._execute_scoped_tools(
            {"messages": [message]},
            {"get_transaction"},
        )

    payloads = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == LOGGER_NAME
    ]

    events = [payload["event"] for payload in payloads]

    assert events == [
        "tool_call_started",
        "tool_call_completed",
    ]

    completed = payloads[1]

    assert completed["tool_name"] == "get_transaction"
    assert completed["success"] is True
    assert completed["duration_ms"] >= 0

    assert "TXN-SECRET" not in caplog.text
    assert result["investigation_data"][0]["result"]["success"]