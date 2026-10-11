
import pytest

from banking_investigator.rag import reranking


class FakeCrossEncoder:
    """Predictable stand-in for the real reranking model."""

    def predict(self, pairs):
        assert len(pairs) == 3
        return [0.1, 0.95, -0.25]


def test_rerank_chunks_sorts_by_score_and_limits_results(
    monkeypatch,
):
    chunks = [
        {"chunk_index": 1, "content": "Address change guidance."},
        {"chunk_index": 2, "content": "Fraud escalation guidance."},
        {"chunk_index": 3, "content": "Review timeline guidance."},
    ]

    monkeypatch.setattr(
        reranking,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder(),
    )

    result = reranking.rerank_chunks(
        "What should staff do if fraud is suspected?",
        chunks,
        top_k=2,
    )

    assert [chunk["chunk_index"] for chunk in result] == [2, 1]
    assert [chunk["rerank_score"] for chunk in result] == [
        0.95,
        0.1,
    ]


def test_rerank_chunks_returns_empty_list_without_loading_model(
    monkeypatch,
):
    def unexpected_model_load():
        raise AssertionError("Model should not load for empty input.")

    monkeypatch.setattr(
        reranking,
        "_get_cross_encoder",
        unexpected_model_load,
    )

    result = reranking.rerank_chunks(
        "What is the policy?",
        [],
    )

    assert result == []


@pytest.mark.parametrize(
    ("query", "chunks", "top_k"),
    [
        ("", [{"chunk_index": 1, "content": "Policy text."}], None),
        ("What is the policy?", [], 0),
    ],
)
def test_rerank_chunks_rejects_invalid_input(query, chunks, top_k):
    with pytest.raises(ValueError):
        reranking.rerank_chunks(
            query,
            chunks,
            top_k=top_k,
        )
