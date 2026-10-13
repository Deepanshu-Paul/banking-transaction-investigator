import json
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, "src")

from banking_investigator.agents import nodes


class GroundingGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal[
        "grounded",
        "insufficient_evidence_handled",
        "unsupported",
    ]
    unsupported_claims: list[str]
    reason: str


EVALUATION_CASES = [
    {
        "id": "G1",
        "request": (
            "What is the amount and status of transaction TXN1001?"
        ),
        "investigation_data": [
            {
                "tool": "get_transaction",
                "result": {
                    "success": True,
                    "data": {
                        "transaction_id": "TXN1001",
                        "amount": "125.00",
                        "currency": "GBP",
                        "status": "completed",
                    },
                    "error": None,
                },
            }
        ],
        "expected_verdict": "grounded",
    },
    {
        "id": "G2",
        "request": (
            "Give me the risk score, customer location, "
            "and device used for TXN1001."
        ),
        "investigation_data": [
            {
                "tool": "get_transaction",
                "result": {
                    "success": True,
                    "data": {
                        "transaction_id": "TXN1001",
                        "status": "declined",
                    },
                    "error": None,
                },
            }
        ],
        "expected_verdict": "insufficient_evidence_handled",
    },
    {
        "id": "G3",
        "request": (
            "Tell me the amount, date, and merchant "
            "for transaction TXN-4040."
        ),
        "investigation_data": [
            {
                "tool": "get_transaction",
                "result": {
                    "success": False,
                    "data": None,
                    "error": "Transaction not found.",
                },
            }
        ],
        "expected_verdict": "insufficient_evidence_handled",
    },
]


def judge_grounding(
    request: str,
    investigation_data: list[dict],
    answer: str,
) -> GroundingGrade:
    payload = {
        "user_request": request,
        "investigation_data": investigation_data,
        "assistant_answer": answer,
    }

    return nodes.llm_client.invoke_structured(
        messages=[
            SystemMessage(
                content=(
                    "You evaluate factual grounding in a banking "
                    "investigation assistant.\n"
                    "Only investigation_data is evidence for factual claims. "
                    "The user's request is not evidence.\n\n"
                    "Choose exactly one verdict using these rules in order:\n\n"
                    "1. unsupported: The answer contains any material factual "
                    "claim not supported by investigation_data. This verdict "
                    "takes priority over the other two.\n\n"
                    "2. insufficient_evidence_handled: No material unsupported "
                    "claims are present, but one or more requested facts cannot "
                    "be established from the evidence, and the assistant clearly "
                    "acknowledges the missing information.\n"
                    "This includes failed lookups where the assistant says "
                    "the record or its details could not be found.\n\n"
                    "3. grounded: The answer addresses all material requested "
                    "facts using supported evidence, with no material unsupported "
                    "claims and no requested facts left unresolved.\n\n"
                    "Important: An answer that accurately reports that a record "
                    "was not found must be classified as "
                    "insufficient_evidence_handled, not grounded, when the "
                    "user requested details that could not be retrieved.\n"
                    "Do not penalize an assistant merely for acknowledging "
                    "that information is unavailable.\n\n"
                    "List any unsupported claims and explain your verdict. "
                    "Return only the required structured evaluation."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            ),
        ],
        output_schema=GroundingGrade,
    )


def evaluate_grounding() -> None:
    passed = 0

    print("\n===== ANSWER GROUNDING EVALUATION =====\n")

    for case in EVALUATION_CASES:
        try:
            result = nodes.final_response_node(
                {
                    "messages": [
                        HumanMessage(content=case["request"])
                    ],
                    "investigation_data": case["investigation_data"],
                }
            )

            answer = result["messages"][-1].content

            grade = judge_grounding(
                request=case["request"],
                investigation_data=case["investigation_data"],
                answer=answer,
            )

            case_passed = (
                grade.verdict == case["expected_verdict"]
            )

            print(
                f"{case['id']} "
                f"[{'PASS' if case_passed else 'FAIL'}] "
                f"Expected: {case['expected_verdict']} | "
                f"Actual: {grade.verdict}"
            )
            print(f"Reason: {grade.reason}")

            if grade.unsupported_claims:
                print(
                    "Unsupported claims: "
                    f"{grade.unsupported_claims}"
                )

            print(f"Answer: {answer[:350]}\n")

        except Exception as exc:
            case_passed = False
            print(
                f"{case['id']} [ERROR] "
                f"{type(exc).__name__}: {exc}\n"
            )

        if case_passed:
            passed += 1

    total = len(EVALUATION_CASES)

    print("===== SUMMARY =====")
    print(f"Cases passed: {passed}/{total}")
    print(f"Evaluation pass rate: {passed / total:.1%}")
    print(
        "Note: The LLM judge is an imperfect evaluator. "
        "Review the answers and verdicts, especially any failures."
    )


if __name__ == "__main__":
    evaluate_grounding()