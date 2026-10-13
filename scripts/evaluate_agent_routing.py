import sys

sys.path.insert(0, "src")

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from banking_investigator.agents.nodes import supervisor_node


EVALUATION_CASES = [
    # Original baseline cases
    {
        "id": "R1",
        "category": "transaction",
        "request": "Investigate transaction TXN1001.",
        "expected": "transaction",
    },
    {
        "id": "R2",
        "category": "transaction",
        "request": "Find transaction TXN2044 and explain its status.",
        "expected": "transaction",
    },
    {
        "id": "R3",
        "category": "account",
        "request": "Show me the details of account ACC1001.",
        "expected": "account",
    },
    {
        "id": "R4",
        "category": "account",
        "request": "What type of account is ACC2002?",
        "expected": "account",
    },
    {
        "id": "R5",
        "category": "policy",
        "request": (
            "What information should staff collect when "
            "a customer disputes a transaction?"
        ),
        "expected": "policy",
    },
    {
        "id": "R6",
        "category": "policy",
        "request": (
            "What is the escalation procedure when staff "
            "suspect card fraud?"
        ),
        "expected": "policy",
    },

    # Additional paraphrased and less direct requests
    {
        "id": "R7",
        "category": "transaction",
        "request": (
            "Can you trace payment reference TXN-7842? "
            "I don't recognise it."
        ),
        "expected": "transaction",
    },
    {
        "id": "R8",
        "category": "transaction",
        "request": (
            "I need to know whether payment TXN-3005 "
            "went through."
        ),
        "expected": "transaction",
    },
    {
        "id": "R9",
        "category": "account",
        "request": "Is account ACC-9021 still active?",
        "expected": "account",
    },
    {
        "id": "R10",
        "category": "account",
        "request": (
            "Tell me the current status and type for ACC-4433."
        ),
        "expected": "account",
    },
    {
        "id": "R11",
        "category": "policy",
        "request": (
            "Before escalating a disputed payment, "
            "what information must staff collect?"
        ),
        "expected": "policy",
    },
    {
        "id": "R12",
        "category": "policy",
        "request": (
            "If a customer may have exposed their card details, "
            "what should staff do next?"
        ),
        "expected": "policy",
    },

    # Simulated multi-agent handoffs
    {
        "id": "H1",
        "category": "handoff",
        "request": "Continue from account lookup to transaction investigation.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Look up account ACC1001 and investigate "
                    "transaction TXN1001 associated with it."
                ),
            },
            {
                "role": "worker",
                "content": (
                    "The account lookup for ACC1001 is complete: "
                    "the account is active. Transaction TXN1001 "
                    "has not been retrieved yet, so the request "
                    "still requires transaction details."
                ),
            },
        ],
        "expected": "transaction",
    },
    {
        "id": "H2",
        "category": "handoff",
        "request": "Continue from transaction lookup to account investigation.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Investigate transaction TXN1001 and tell me "
                    "whether account ACC1001 is active."
                ),
            },
            {
                "role": "worker",
                "content": (
                    "The transaction lookup for TXN1001 is complete. "
                    "It is associated with account ACC1001. The "
                    "account's current status has not been checked."
                ),
            },
        ],
        "expected": "account",
    },
]


def build_messages(case: dict) -> list:
    if "messages" not in case:
        return [
            HumanMessage(content=case["request"])
        ]

    messages = []

    for item in case["messages"]:
        if item["role"] == "user":
            messages.append(
                HumanMessage(content=item["content"])
            )
        elif item["role"] == "worker":
            messages.append(
                AIMessage(content=item["content"])
            )
        else:
            raise ValueError(
                f"Unsupported message role: {item['role']}"
            )

    return messages


def evaluate_routing() -> None:
    correct = 0
    results = []

    print("\n===== AGENT ROUTING EVALUATION =====\n")

    for case in EVALUATION_CASES:
        state = {
            "messages": build_messages(case),
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
                "category": case["category"],
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

    total = len(results)
    accuracy = correct / total if total else 0.0

    print("\n===== OVERALL SUMMARY =====")
    print(f"Correct routes: {correct}/{total}")
    print(f"Routing accuracy: {accuracy:.1%}")

    print("\n===== CATEGORY SUMMARY =====")

    for category in (
        "transaction",
        "account",
        "policy",
        "handoff",
    ):
        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        if not category_results:
            continue

        category_correct = sum(
            result["passed"]
            for result in category_results
        )

        print(
            f"{category}: "
            f"{category_correct}/{len(category_results)} "
            f"correct"
        )

    failures = [
        result
        for result in results
        if not result["passed"]
    ]

    if failures:
        print("\n===== CASES TO INVESTIGATE =====")

        for result in failures:
            print(
                f"{result['id']}: "
                f"expected={result['expected']}, "
                f"actual={result['actual']}"
            )

    print(
        "\nNote: This is a small, manually labelled dataset. "
        "The handoff cases simulate conversation history; "
        "they do not execute the complete agent graph."
    )


if __name__ == "__main__":
    evaluate_routing()