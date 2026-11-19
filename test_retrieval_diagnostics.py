from app.services.semantic_search_service import SemanticSearchService
from app.services.bm25_search_service import BM25SearchService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUESTIONS = [
    "What year was the speaker born?",
    "Which university did the speaker graduate from?",
    "What is the speaker's home address?"
]


print("Initializing retrieval services...")

semantic_service = SemanticSearchService()
bm25_service = BM25SearchService()


for question in QUESTIONS:

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    semantic_results = semantic_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=5
    )

    bm25_results = bm25_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=5
    )

    print("\nSEMANTIC SEARCH RESULTS")

    for result in semantic_results:
        print(
            f"\nChunk: {result['chunk_index']}"
            f"\nSimilarity: {result['similarity']:.4f}"
            f"\nText: {result['text'][:250]}"
        )

    print("\nBM25 SEARCH RESULTS")

    for result in bm25_results:
        print(
            f"\nChunk: {result['chunk_index']}"
            f"\nBM25 Score: {result['bm25_score']:.4f}"
            f"\nText: {result['text'][:250]}"
        )

print("\nRetrieval diagnostics completed.")