from datetime import datetime, timezone

from banking_investigator.rag import ingestion


def test_ingest_pdf_connects_loading_chunking_and_storage(
    monkeypatch,
) -> None:
    content = (
        "Banking policy: disputed transactions must be investigated. "
        * 40
    )

    saved_documents = []
    saved_chunks = []

    class FakeRepository:
        def create_document(self, **kwargs):
            saved_documents.append(kwargs)

        def create_chunk(self, **kwargs):
            saved_chunks.append(kwargs)
            return f"chunk-{kwargs['chunk_index']}"

    monkeypatch.setattr(
        ingestion,
        "load_pdf_text",
        lambda _path: content,
    )
    monkeypatch.setattr(
        ingestion,
        "embed_text",
        lambda _text: [0.1] * 1536,
    )
    monkeypatch.setattr(
        ingestion,
        "RagRepository",
        FakeRepository,
    )

    result = ingestion.ingest_pdf(
        pdf_path="sample.pdf",
        document_id="DISPUTES-001",
        version=1,
        title="Disputed Transactions Policy",
        source="sample.pdf",
        document_type="policy",
        status="active",
        effective_from=datetime.now(timezone.utc),
        effective_to=None,
        metadata={"region": "UK"},
        max_tokens=50,
        overlap_tokens=10,
    )

    assert result["document_id"] == "DISPUTES-001"
    assert len(saved_documents) == 1
    assert saved_documents[0]["content"] == content
    assert len(saved_chunks) > 1
    assert all(
        len(chunk["embedding"]) == 1536
        for chunk in saved_chunks
    )