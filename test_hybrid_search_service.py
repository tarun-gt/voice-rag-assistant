
from app.services.hybrid_search_service import HybridSearchService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUERIES = [
    "What does the speaker say about self-reflection?",
    "Why should teachers admit when they are wrong?",
    "How does a teacher demonstrate humility to students?",
    "What are the three most powerful words a teacher can say?"
]


print("Initializing hybrid search service...")

hybrid_search_service = HybridSearchService()


for query_number, query in enumerate(QUERIES, start=1):

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}: {query}")
    print("=" * 80)

    results = hybrid_search_service.search(
        document_id=DOCUMENT_ID,
        query=query,
        top_k=3
    )

    # 1. Verify the number of returned results
    assert len(results) == 3

    print("\nTop 3 hybrid results retrieved successfully!")

    # 2. Verify RRF scores are in descending order
    scores = [
        chunk["rrf_score"]
        for chunk in results
    ]

    reranker_scores = [
        result["reranker_score"]
        for result in results
    ]

    assert reranker_scores == sorted(
        reranker_scores,
        reverse=True
    )

    print("Hybrid ranking validation passed!")

    # 3. Verify the required chunk fields
    for chunk in results:

        assert chunk["document_id"] == DOCUMENT_ID
        assert "chunk_index" in chunk
        assert "text" in chunk
        assert "start" in chunk
        assert "end" in chunk
        assert "rrf_score" in chunk

    print("Retrieved chunk structure validation passed!")

    # 4. Verify that no duplicate chunks are returned
    chunk_indexes = [
        chunk["chunk_index"]
        for chunk in results
    ]

    assert len(chunk_indexes) == len(set(chunk_indexes))

    print("Duplicate chunk validation passed!")

    # 5. Display results
    for rank, chunk in enumerate(results, start=1):

        print("\n" + "-" * 70)

        print(f"Rank: {rank}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"RRF score: {chunk['rrf_score']:.6f}")
        print(
            f"Timestamp: {chunk['start']}s - "
            f"{chunk['end']}s"
        )
        print(f"Text: {chunk['text']}")


print("\nAll hybrid search experiments completed successfully!")
