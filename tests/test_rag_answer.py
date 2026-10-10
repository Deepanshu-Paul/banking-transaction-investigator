
from langchain_core.messages import AIMessage

from banking_investigator.rag import answer
from banking_investigator.rag.grading import ChunkRelevanceGrade


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

    # Mock retrieval so the test does not query the database.
    monkeypatch.setattr(
        answer,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: retrieved_chunks,
    )

    # Mock the answer-generation LLM.
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

    # Mock the relevance grader.
    monkeypatch.setattr(
        answer,
        "grade_chunk_relevance",
        lambda query, chunk_content: ChunkRelevanceGrade(
            verdict="relevant",
            reason="The chunk contains useful evidence.",
        ),
    )

    result = answer.answer_with_rag(
        "How should an unrecognised transaction be handled?",
        document_type="policy",
        region="UK",
    )

    # Verify the answer uses the retrieved policy context.
    assert "route suspected fraud" in result["answer"]
    assert "[Source 1]" in result["answer"]

    # Verify source metadata is preserved.
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "DISPUTES-001"

    # Verify the retrieved context was sent to the LLM.
    assert any(
        "unrecognised transaction" in str(message.content).lower()
        for message in captured_messages
    )

def test_answer_with_rag_skips_generation_when_all_chunks_irrelevant(
    monkeypatch,
) -> None:
    retrieved_chunks = [
        {
            "title": "Address Change Policy",
            "document_id": "ADDRESS-001",
            "document_version": 1,
            "chunk_index": 0,
            "content": "Customers can update their registered address.",
            "metadata": {"source": "address_policy.pdf"},
            "similarity": 0.65,
        }
    ]

    monkeypatch.setattr(
        answer,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: retrieved_chunks,
    )

    monkeypatch.setattr(
        answer,
        "grade_chunk_relevance",
        lambda query, chunk_content: ChunkRelevanceGrade(
            verdict="irrelevant",
            reason="The chunk does not address disputed transactions.",
        ),
    )

    monkeypatch.setattr(
    answer,
    "rewrite_query",
    lambda query, relevance_feedback: (
        "Bank policy for unrecognised transaction handling"
        ),
    )

    # Fail the test if answer generation is attempted.
    def unexpected_invoke(*args, **kwargs):
        raise AssertionError(
            "LLM answer generation should not run without relevant evidence."
        )

    monkeypatch.setattr(
        answer.llm_client,
        "invoke",
        unexpected_invoke,
    )

    result = answer.answer_with_rag(
        "How should staff handle an unrecognised transaction?",
        document_type="policy",
        region="UK",
    )

    assert (
        "sufficiently relevant policy evidence"
        in result["answer"]
    )
    assert result["sources"] == []


def test_answer_with_rag_recovers_after_query_rewriting(
    monkeypatch,
) -> None:
    original_query = (
        "How should staff handle an unrecognised transaction?"
    )
    rewritten_query = (
        "Bank policy for suspected fraud and card-security escalation"
    )

    irrelevant_chunk = {
        "title": "Address Change Policy",
        "document_id": "ADDRESS-001",
        "document_version": 1,
        "chunk_index": 0,
        "content": "Customers can update their registered address.",
        "metadata": {"source": "address_policy.pdf"},
        "similarity": 0.65,
    }

    relevant_chunk = {
        "title": "Disputed Transactions Policy",
        "document_id": "DISPUTES-001",
        "document_version": 1,
        "chunk_index": 2,
        "content": (
            "Route suspected fraud to the card-security "
            "review queue."
        ),
        "metadata": {"source": "sample_policy.pdf"},
        "similarity": 0.78,
    }

    retrieval_queries = []
    generation_calls = []

    def fake_retrieve(query, **kwargs):
        retrieval_queries.append(query)

        if query == original_query:
            return [irrelevant_chunk]

        if query == rewritten_query:
            return [relevant_chunk]

        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(
        answer,
        "retrieve_relevant_chunks",
        fake_retrieve,
    )

    monkeypatch.setattr(
        answer,
        "grade_chunk_relevance",
        lambda query, chunk_content: ChunkRelevanceGrade(
            verdict=(
                "relevant"
                if "card-security" in chunk_content
                else "irrelevant"
            ),
            reason=(
                "The text describes fraud escalation."
                if "card-security" in chunk_content
                else "The text discusses address changes."
            ),
        ),
    )

    monkeypatch.setattr(
        answer,
        "rewrite_query",
        lambda query, relevance_feedback: rewritten_query,
    )

    def fake_invoke(messages, tools=None):
        generation_calls.append(messages)
        return AIMessage(
            content=(
                "Route suspected fraud to the card-security "
                "review queue. [Source 1]"
            )
        )

    monkeypatch.setattr(
        answer.llm_client,
        "invoke",
        fake_invoke,
    )

    result = answer.answer_with_rag(
        original_query,
        document_type="policy",
        region="UK",
    )

    assert retrieval_queries == [original_query, rewritten_query]
    assert len(generation_calls) == 1
    assert "card-security review queue" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "DISPUTES-001"
