from datetime import datetime
from typing import Any
from pathlib import Path

from banking_investigator.rag.pdf_loader import load_pdf_text
from banking_investigator.rag.chunking import chunk_text
from banking_investigator.rag.embeddings import embed_text
from banking_investigator.repositories.rag_repository import RagRepository


def ingest_document(
    *,
    document_id: str,
    version: int,
    title: str,
    source: str,
    document_type: str,
    status: str,
    effective_from: datetime,
    effective_to: datetime | None,
    content: str,
    metadata: dict[str, Any] | None = None,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> dict[str, str | int]:
    if not content.strip():
        raise ValueError("Document content must not be empty")

    if version < 1:
        raise ValueError("Document version must be >= 1")

    if effective_to is not None and effective_to <= effective_from:
        raise ValueError(
            "effective_to must be later than effective_from"
        )

    document_metadata = metadata or {}

    chunks = chunk_text(
        content,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Generate embeddings before writing anything to the database.
    embedded_chunks = [
        (chunk, embed_text(chunk.content))
        for chunk in chunks
    ]

    for _, embedding in embedded_chunks:
        if len(embedding) != 1536:
            raise ValueError(
                "Embedding must contain exactly 1536 dimensions"
            )

    repository = RagRepository()

    repository.create_document(
        document_id=document_id,
        version=version,
        title=title,
        source=source,
        document_type=document_type,
        status=status,
        effective_from=effective_from,
        effective_to=effective_to,
        content=content,
        metadata=document_metadata,
    )

    for chunk, embedding in embedded_chunks:
        chunk_metadata = {
            **document_metadata,
            "document_type": document_type,
            "status": status,
            "source": source,
        }

        repository.create_chunk(
            document_id=document_id,
            document_version=version,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            token_count=chunk.token_count,
            embedding=embedding,
            metadata=chunk_metadata,
        )

    return {
        "document_id": document_id,
        "version": version,
        "chunk_count": len(chunks),
    }

def ingest_pdf(
    *,
    pdf_path: str | Path,
    document_id: str,
    version: int,
    title: str,
    source: str,
    document_type: str,
    status: str,
    effective_from: datetime,
    effective_to: datetime | None,
    metadata: dict[str, Any] | None = None,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> dict[str, str | int]:
    content = load_pdf_text(pdf_path)

    return ingest_document(
        document_id=document_id,
        version=version,
        title=title,
        source=source,
        document_type=document_type,
        status=status,
        effective_from=effective_from,
        effective_to=effective_to,
        content=content,
        metadata=metadata,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )