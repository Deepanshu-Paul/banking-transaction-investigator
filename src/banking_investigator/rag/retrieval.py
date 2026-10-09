from typing import Any

from banking_investigator.rag.embeddings import embed_text
from banking_investigator.repositories.rag_repository import RagRepository


def retrieve_relevant_chunks(
    query: str,
    *,
    top_k: int = 3,
    document_type: str | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    query_embedding = embed_text(query)

    if len(query_embedding) != 1536:
        raise ValueError(
            "Query embedding must contain 1536 dimensions"
        )

    repository = RagRepository()

    return repository.search_similar_chunks(
        query_embedding,
        top_k=top_k,
        document_type=document_type,
        region=region,
    )