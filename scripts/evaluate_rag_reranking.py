
from sentence_transformers import CrossEncoder

from banking_investigator.rag.retrieval import retrieve_relevant_chunks


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 8
FINAL_K = 3

EVALUATION_CASES = [
    {
        "query": (
            "What steps should staff take when a customer "
            "reports an unrecognised transaction?"
        ),
        "relevant_chunks": {1, 2, 5, 7},
    },
    {
        "query": (
            "What information should staff collect "
            "for a disputed transaction?"
        ),
        "relevant_chunks": {1, 2, 3, 4},
    },
    {
        "query": "What should staff do if fraud is suspected?",
        "relevant_chunks": {2, 5, 7},
    },
]


def recall_at_k(results, relevant_chunks):
    hits = sum(
        chunk["chunk_index"] in relevant_chunks
        for chunk in results[:FINAL_K]
    )
    return hits / len(relevant_chunks)


def reciprocal_rank(results, relevant_chunks):
    for rank, chunk in enumerate(results, start=1):
        if chunk["chunk_index"] in relevant_chunks:
            return 1 / rank

    return 0.0


def rerank_chunks(model, query, candidates):
    pairs = [
        (query, chunk["content"])
        for chunk in candidates
    ]

    scores = model.predict(pairs)

    scored_chunks = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in zip(candidates, scores)
    ]

    return sorted(
        scored_chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )


def main():
    print(f"Loading reranker: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME)
    print("Reranker loaded.")

    baseline_recalls = []
    reranked_recalls = []
    baseline_rrs = []
    reranked_rrs = []

    for case in EVALUATION_CASES:
        query = case["query"]
        relevant = case["relevant_chunks"]

        candidates = retrieve_relevant_chunks(
            query,
            top_k=CANDIDATE_K,
            document_type="policy",
            region="UK",
        )

        baseline = candidates[:FINAL_K]

        reranked = rerank_chunks(
            model,
            query,
            candidates,
        )[:FINAL_K]

        baseline_recall = recall_at_k(baseline, relevant)
        reranked_recall = recall_at_k(reranked, relevant)

        baseline_rr = reciprocal_rank(baseline, relevant)
        reranked_rr = reciprocal_rank(reranked, relevant)

        baseline_recalls.append(baseline_recall)
        reranked_recalls.append(reranked_recall)
        baseline_rrs.append(baseline_rr)
        reranked_rrs.append(reranked_rr)

        print(f"\nQUERY: {query}")

        print("\nOriginal vector ranking:")
        for rank, chunk in enumerate(baseline, start=1):
            label = (
                "RELEVANT"
                if chunk["chunk_index"] in relevant
                else "NOT LABELLED RELEVANT"
            )
            print(
                f"Rank {rank}: chunk {chunk['chunk_index']} | "
                f"similarity={chunk['similarity']:.3f} | "
                f"{label}"
            )

        print("\nCross-Encoder reranking:")
        for rank, chunk in enumerate(reranked, start=1):
            label = (
                "RELEVANT"
                if chunk["chunk_index"] in relevant
                else "NOT LABELLED RELEVANT"
            )
            print(
                f"Rank {rank}: chunk {chunk['chunk_index']} | "
                f"score={chunk['rerank_score']:.3f} | "
                f"{label}"
            )

        print(
            f"\nRecall@{FINAL_K}: "
            f"{baseline_recall:.3f} -> {reranked_recall:.3f}"
        )
        print(
            f"Reciprocal rank: "
            f"{baseline_rr:.3f} -> {reranked_rr:.3f}"
        )

    print("\n" + "=" * 45)
    print(
        f"Mean Recall@{FINAL_K}: "
        f"{sum(baseline_recalls) / len(baseline_recalls):.3f}"
        f" -> "
        f"{sum(reranked_recalls) / len(reranked_recalls):.3f}"
    )
    print(
        f"MRR: "
        f"{sum(baseline_rrs) / len(baseline_rrs):.3f}"
        f" -> "
        f"{sum(reranked_rrs) / len(reranked_rrs):.3f}"
    )


if __name__ == "__main__":
    main()
