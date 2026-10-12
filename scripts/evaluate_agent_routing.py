import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from banking_investigator.agents.nodes import supervisor_node


# Small, manually labelled baseline evaluation dataset.
EVALUATION_CASES = [
    {
        "id": "R1",
        "request": "Investigate transaction TXN1001.",
        "expected": "transaction",
    },
    {
        "id": "R2",
        "request": "Find transaction TXN2044 and explain its status.",
        "expected": "transaction",
    },
    {
        "id": "R3",
        "request": "Show me the details of account ACC1001.",
        "expected": "account",
    },
    {
        "id": "R4",
        "request": "What type of account is ACC2002?",
        "expected": "account",
    },
    {
        "id": "R5",
        "request": (
            "What information should staff collect when "
            "a customer disputes a transaction?"
        ),
        "expected": "policy",
    },
    {
        "id": "R6",
        "request": (
            "What is the escalation procedure when staff "
            "suspect card fraud?"
        ),
        "expected": "policy",
    },
]


def evaluate_routing() -> None:
    correct = 0
    results = []

    print("\n===== AGENT ROUTING EVALUATION =====\n")

    for case in EVALUATION_CASES:
        state = {
            "messages": [
                HumanMessage(content=case["request"])
            ],
            "customer_id": None,
        }

        try:
            result = supervisor_node(
                state,
                Runtime(store=None),
            )
            actual = result["next_agent"]
            passed = actual == case["expected"]

        except Exception as exc:
            actual = f"ERROR: {type(exc).__name__}"
            passed = False

        if passed:
            correct += 1

        results.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "actual": actual,
                "passed": passed,
            }
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"{case['id']} [{status}] "
            f"Expected: {case['expected']} | "
            f"Actual: {actual}"
        )

    total = len(EVALUATION_CASES)
    accuracy = correct / total

    print("\n===== SUMMARY =====")
    print(f"Correct routes: {correct}/{total}")
    print(f"Routing accuracy: {accuracy:.1%}")

    failures = [
        result for result in results
        if not result["passed"]
    ]

    if failures:
        print("\nCases to investigate:")
        for result in failures:
            print(
                f"{result['id']}: "
                f"expected={result['expected']}, "
                f"actual={result['actual']}"
            )

    print(
        "\nNote: This is a small, manually labelled "
        "baseline, not proof of general routing accuracy."
    )


if __name__ == "__main__":
    evaluate_routing()