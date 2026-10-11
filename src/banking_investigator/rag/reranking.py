
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Load the model once and reuse it."""
    return CrossEncoder(MODEL_NAME)


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    if not chunks:
        return []

    model = _get_cross_encoder()

    pairs = [
        (query, chunk["content"])
        for chunk in chunks
    ]

    scores = model.predict(pairs)

    if len(scores) != len(chunks):
        raise RuntimeError(
            "Reranker returned an unexpected number of scores."
        )

    reranked = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in zip(chunks, scores)
    ]

    reranked.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    if top_k is not None:
        return reranked[:top_k]

    return reranked
