
from banking_investigator.rag.retrieval import retrieve_relevant_chunks


TOP_K = 3

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


def main() -> None:
    recall_scores = []
    reciprocal_ranks = []

    for case in EVALUATION_CASES:
        query = case["query"]
        relevant = case["relevant_chunks"]

        results = retrieve_relevant_chunks(
            query,
            top_k=TOP_K,
            document_type="policy",
            region="UK",
        )

        retrieved = [
            chunk["chunk_index"] for chunk in results
        ]

        relevant_hits = sum(
            chunk_index in relevant
            for chunk_index in retrieved
        )

        recall = relevant_hits / len(relevant)

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_index in enumerate(
                    retrieved, start=1
                )
                if chunk_index in relevant
            ),
            None,
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQUERY: {query}")
        print(f"Expected relevant chunks: {sorted(relevant)}")

        for rank, chunk in enumerate(results, start=1):
            is_relevant = (
                chunk["chunk_index"] in relevant
            )
            label = "RELEVANT" if is_relevant else "NOT LABELLED RELEVANT"

            print(
                f"Rank {rank}: chunk {chunk['chunk_index']} | "
                f"similarity={chunk['similarity']:.3f} | "
                f"{label}"
            )

        print(f"Recall@{TOP_K}: {recall:.3f}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")

    mean_recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 45)
    print(f"Mean Recall@{TOP_K}: {mean_recall:.3f}")
    print(f"MRR:                {mrr:.3f}")


if __name__ == "__main__":
    main()
