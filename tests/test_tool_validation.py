import pytest
from pydantic import ValidationError

from banking_investigator.agents import tool_executor
from banking_investigator.agents.tool_validation import (
    validate_tool_arguments,
)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected"),
    [
        (
            "get_transaction",
            {"transaction_id": "TXN-123"},
            {"transaction_id": "TXN-123"},
        ),
        (
            "get_account",
            {"account_id": "ACC-123"},
            {"account_id": "ACC-123"},
        ),
        (
            "get_account",
            {"account_id": "  ACC-123  "},
            {"account_id": "ACC-123"},
        ),
    ],
)
def test_valid_tool_arguments(
    tool_name,
    arguments,
    expected,
):
    assert validate_tool_arguments(
        tool_name,
        arguments,
    ) == expected


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("get_transaction", {}),
        ("get_transaction", {"transaction_id": ""}),
        ("get_transaction", {"transaction_id": "   "}),
        ("get_transaction", {"transaction_id": 123}),
        (
            "get_transaction",
            {
                "transaction_id": "TXN-123",
                "account_id": "ACC-123",
            },
        ),
        (
            "get_account",
            {
                "account_id": "ACC-123",
                "unexpected": "value",
            },
        ),
    ],
)
def test_invalid_tool_arguments(
    tool_name,
    arguments,
):
    with pytest.raises(ValidationError):
        validate_tool_arguments(
            tool_name,
            arguments,
        )


def test_unknown_tool_has_no_argument_schema():
    with pytest.raises(ValueError):
        validate_tool_arguments(
            "delete_transaction",
            {},
        )


def test_executor_rejects_invalid_arguments_before_execution(
    monkeypatch,
):
    called = False

    def fake_tool(transaction_id, deadline=None):
        nonlocal called
        called = True
        return {"transaction_id": transaction_id}

    monkeypatch.setitem(
        tool_executor.TOOL_REGISTRY,
        "get_transaction",
        fake_tool,
    )

    result = tool_executor.execute_tool(
        "get_transaction",
        {
            "transaction_id": "TXN-123",
            "unexpected": "value",
        },
    )

    assert result.success is False
    assert result.error == (
        "Invalid arguments for tool 'get_transaction'."
    )
    assert called is False