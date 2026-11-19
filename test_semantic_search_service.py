
from app.services.semantic_search_service import SemanticSearchService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUERIES = [
    "What does the speaker say about self-reflection?",
    "Why should teachers admit when they are wrong?",
    "How does a teacher demonstrate humility to students?",
    "What are the three most powerful words a teacher can say?"
]


print("Initializing semantic search service...")

semantic_search_service = SemanticSearchService()


for query_number, query in enumerate(QUERIES, start=1):

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}: {query}")
    print("=" * 80)

    results = semantic_search_service.search(
        document_id=DOCUMENT_ID,
        query=query,
        top_k=3
    )

    # 1. Verify the number of returned results
    assert len(results) == 3

    print("\nTop 3 results retrieved successfully!")

    # 2. Verify similarity scores are in descending order
    similarity_scores = [
        chunk["similarity"]
        for chunk in results
    ]

    assert similarity_scores == sorted(
        similarity_scores,
        reverse=True
    )

    print("Similarity ranking validation passed!")

    # 3. Verify the required chunk fields
    for chunk in results:

        assert chunk["document_id"] == DOCUMENT_ID
        assert "chunk_index" in chunk
        assert "text" in chunk
        assert "start" in chunk
        assert "end" in chunk
        assert "embedding" in chunk
        assert "similarity" in chunk

    print("Retrieved chunk structure validation passed!")

    # 4. Display search results
    for rank, chunk in enumerate(results, start=1):

        print("\n" + "-" * 70)

        print(f"Rank: {rank}")
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(
            f"Timestamp: {chunk['start']}s - "
            f"{chunk['end']}s"
        )
        print(f"Text: {chunk['text']}")


print("\nAll semantic search experiments completed successfully!")
