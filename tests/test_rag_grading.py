
import pytest

from banking_investigator.rag import grading


def test_grade_chunk_relevance_returns_structured_result(monkeypatch):
    expected = grading.ChunkRelevanceGrade(
        verdict="relevant",
        reason="The text explains how to handle unrecognised transactions.",
    )

    captured = {}

    def fake_invoke_structured(schema, messages):
        captured["schema"] = schema
        captured["messages"] = messages
        return expected

    monkeypatch.setattr(
        grading.llm_client,
        "invoke_structured",
        fake_invoke_structured,
    )

    result = grading.grade_chunk_relevance(
        "How should staff handle an unrecognised transaction?",
        "Route the case to card-security review if fraud is suspected.",
    )

    assert result == expected
    assert captured["schema"] is grading.ChunkRelevanceGrade
    assert len(captured["messages"]) == 2


@pytest.mark.parametrize(
    ("query", "chunk"),
    [
        ("", "Some policy text"),
        ("What is the procedure?", ""),
    ],
)
def test_grade_chunk_relevance_rejects_empty_input(query, chunk):
    with pytest.raises(ValueError):
        grading.grade_chunk_relevance(query, chunk)
