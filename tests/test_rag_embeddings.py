from banking_investigator.rag import embeddings


def test_embed_text(monkeypatch) -> None:
    monkeypatch.setattr(
        type(embeddings.embedding_model),
        "embed_query",
        lambda self, _text: [0.1] * 1536,
    )

    result = embeddings.embed_text(
        "Banking policy test."
    )

    assert len(result) == 1536
    assert result[0] == 0.1