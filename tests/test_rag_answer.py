from langchain_core.messages import AIMessage

from banking_investigator.rag import answer


def test_answer_with_rag_uses_retrieved_context(
    monkeypatch,
) -> None:
    retrieved_chunks = [
        {
            "title": "Disputed Transactions Policy",
            "document_id": "DISPUTES-001",
            "document_version": 1,
            "chunk_index": 2,
            "content": (
                "For an unrecognised transaction, "
                "confirm the transaction details and "
                "route suspected fraud for review."
            ),
            "metadata": {
                "source": "sample_policy.pdf",
            },
            "similarity": 0.82,
        }
    ]

    captured_messages = []

    monkeypatch.setattr(
        answer,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: retrieved_chunks,
    )

    def fake_invoke(messages, tools=None):
        captured_messages.extend(messages)
        return AIMessage(
            content=(
                "Confirm the transaction details and "
                "route suspected fraud for review. [Source 1]"
            )
        )

    monkeypatch.setattr(
        answer.llm_client,
        "invoke",
        fake_invoke,
    )

    result = answer.answer_with_rag(
        "How should an unrecognised transaction be handled?",
        document_type="policy",
        region="UK",
    )

    assert "route suspected fraud" in result["answer"]
    assert "[Source 1]" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "DISPUTES-001"
    assert any(
        "unrecognised transaction" in str(message.content).lower()
        for message in captured_messages
    )