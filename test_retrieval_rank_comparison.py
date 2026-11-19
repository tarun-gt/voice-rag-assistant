from app.services.semantic_search_service import SemanticSearchService
from app.services.bm25_search_service import BM25SearchService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUESTIONS = [
    "Which university does the speaker say they attended?",
    "What was the speaker's exact graduation year?"
]

semantic_service = SemanticSearchService()
bm25_service = BM25SearchService()

for question in QUESTIONS:

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    semantic_results = semantic_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=10
    )

    bm25_results = bm25_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=10
    )

    print("\nSEMANTIC TOP 10")

    for rank, result in enumerate(semantic_results, start=1):
        print(
            f"\nRank: {rank}"
            f"\nChunk: {result['chunk_index']}"
            f"\nScore: {result['similarity']:.4f}"
            f"\nText: {result['text'][:200]}"
        )

    print("\nBM25 TOP 10")

    for rank, result in enumerate(bm25_results, start=1):
        print(
            f"\nRank: {rank}"
            f"\nChunk: {result['chunk_index']}"
            f"\nScore: {result['bm25_score']:.4f}"
            f"\nText: {result['text'][:200]}"
        )

print("\nRank comparison completed.")