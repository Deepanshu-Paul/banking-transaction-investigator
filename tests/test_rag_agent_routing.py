from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from banking_investigator.agents import nodes
from banking_investigator.agents.routing import SupervisorDecision


def test_supervisor_can_route_policy_questions(
    monkeypatch,
) -> None:
    captured_messages = []

    def fake_invoke_structured(messages, output_schema):
        captured_messages.extend(messages)
        assert output_schema is SupervisorDecision
        return SupervisorDecision(next_agent="policy")

    monkeypatch.setattr(
        nodes.llm_client,
        "invoke_structured",
        fake_invoke_structured,
    )

    result = nodes.supervisor_node(
        {
            "messages": [
                HumanMessage(
                    content="How should an unrecognised transaction be handled?"
                )
            ],
            "customer_id": None,
        },
        Runtime(store=None),
    )

    assert result["next_agent"] == "policy"

    prompt = captured_messages[0].content
    assert "policy" in prompt.lower()
    assert "using RAG" in prompt