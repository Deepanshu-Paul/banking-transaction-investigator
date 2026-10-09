from banking_investigator.rag.chunking import chunk_text


def test_chunk_text_splits_document() -> None:
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(
        text,
        max_tokens=20,
        overlap_tokens=5,
    )

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert all(
        chunk.token_count <= 20
        for chunk in chunks
    )