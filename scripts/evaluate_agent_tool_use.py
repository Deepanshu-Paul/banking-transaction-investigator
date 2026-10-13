import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage

from banking_investigator.agents.nodes import (
    account_agent_node,
    transaction_agent_node,
)
from banking_investigator.agents.tool_validation import (
    validate_tool_arguments,
)


EVALUATION_CASES = [
    {
        "id": "T1",
        "agent": "transaction",
        "request": (
            "Investigate transaction TXN1001 "
            "and provide its details."
        ),
        "expected_tool": "get_transaction",
        "expected_arguments": {"transaction_id": "TXN1001"},
    },
    {
        "id": "T2",
        "agent": "transaction",
        "request": (
            "I don't recognise payment TXN-7842. "
            "Can you look it up?"
        ),
        "expected_tool": "get_transaction",
        "expected_arguments": {"transaction_id": "TXN-7842"},
    },
    {
        "id": "T3",
        "agent": "transaction",
        "request": "What happened with transaction TXN2044?",
        "expected_tool": "get_transaction",
        "expected_arguments": {"transaction_id": "TXN2044"},
    },
    {
        "id": "A1",
        "agent": "account",
        "request": "Show me the details of account ACC1001.",
        "expected_tool": "get_account",
        "expected_arguments": {"account_id": "ACC1001"},
    },
    {
        "id": "A2",
        "agent": "account",
        "request": "Is account ACC-9021 currently active?",
        "expected_tool": "get_account",
        "expected_arguments": {"account_id": "ACC-9021"},
    },
    {
        "id": "A3",
        "agent": "account",
        "request": (
            "Tell me the account type and status for ACC-4433."
        ),
        "expected_tool": "get_account",
        "expected_arguments": {"account_id": "ACC-4433"},
    },
]


AGENTS = {
    "transaction": transaction_agent_node,
    "account": account_agent_node,
}


def evaluate_tool_use() -> None:
    passed_count = 0
    results = []

    print("\n===== AGENT TOOL-USE EVALUATION =====\n")

    for case in EVALUATION_CASES:
        try:
            agent_node = AGENTS[case["agent"]]

            # Call the agent, but do not execute its requested tool.
            result = agent_node(
                {
                    "messages": [
                        HumanMessage(content=case["request"])
                    ]
                }
            )

            response = result["messages"][-1]
            tool_calls = response.tool_calls

            if len(tool_calls) != 1:
                actual = (
                    "No tool call"
                    if not tool_calls
                    else f"{len(tool_calls)} tool calls"
                )
                passed = False
            else:
                tool_call = tool_calls[0]
                actual_tool = tool_call["name"]
                actual_args = tool_call["args"]

                try:
                    validated_args = validate_tool_arguments(
                        actual_tool,
                        actual_args,
                    )
                    arguments_valid = (
                        validated_args == case["expected_arguments"]
                    )
                except (ValueError, Exception):
                    arguments_valid = False

                passed = (
                    actual_tool == case["expected_tool"]
                    and arguments_valid
                )

                actual = (
                    f"{actual_tool}({actual_args})"
                )

        except Exception as exc:
            passed = False
            actual = f"ERROR: {type(exc).__name__}"

        if passed:
            passed_count += 1

        results.append(
            {
                "id": case["id"],
                "category": case["agent"],
                "passed": passed,
            }
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"{case['id']} [{status}] "
            f"Expected: {case['expected_tool']}"
            f"({case['expected_arguments']}) | "
            f"Actual: {actual}"
        )

    total = len(results)

    print("\n===== SUMMARY =====")
    print(f"Correct tool calls: {passed_count}/{total}")
    print(
        "Tool-use accuracy: "
        f"{passed_count / total:.1%}"
    )

    for category in ("transaction", "account"):
        cases = [
            result
            for result in results
            if result["category"] == category
        ]
        correct = sum(result["passed"] for result in cases)

        print(f"{category}: {correct}/{len(cases)} correct")

    print(
        "\nNote: This script checks tool selection and arguments. "
        "It does not execute banking tools or query the database."
    )


if __name__ == "__main__":
    evaluate_tool_use()