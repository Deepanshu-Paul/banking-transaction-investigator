
import pytest

from banking_investigator.rag import query_rewriting


def test_rewrite_query_returns_structured_rewritten_query(
    monkeypatch,
):
    expected = query_rewriting.QueryRewrite(
        rewritten_query=(
            "Bank policy for suspected fraudulent card "
            "transactions and security escalation"
        ),
        reason="Focuses on the fraud-handling procedure.",
    )

    captured = {}

    def fake_invoke_structured(schema, messages):
        captured["schema"] = schema
        captured["messages"] = messages
        return expected

    monkeypatch.setattr(
        query_rewriting.llm_client,
        "invoke_structured",
        fake_invoke_structured,
    )

    result = query_rewriting.rewrite_query(
        "How should staff handle an unrecognised transaction?",
        [
            "The retrieved chunk discusses address changes.",
            "The retrieved chunk does not explain transaction disputes.",
        ],
    )

    assert result == expected.rewritten_query
    assert captured["schema"] is query_rewriting.QueryRewrite
    assert len(captured["messages"]) == 2

    prompt_text = "\n".join(
        str(message.content)
        for message in captured["messages"]
    )
    assert "Original question" in prompt_text
    assert "address changes" in prompt_text


@pytest.mark.parametrize(
    ("query", "feedback"),
    [
        ("", ["No relevant evidence found"]),
        ("What is the procedure?", []),
    ],
)
def test_rewrite_query_rejects_invalid_input(query, feedback):
    with pytest.raises(ValueError):
        query_rewriting.rewrite_query(query, feedback)
