from app.services.semantic_search_service import SemanticSearchService
from app.services.bm25_search_service import BM25SearchService


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

TEST_CASES = [
    {
        "question": "What are the three most powerful words a teacher can say?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "Why should teachers admit when they are wrong?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "Which university does the speaker say they attended?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "What year was the speaker born?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "What is the speaker's home address?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "What was the speaker's exact graduation year?",
        "expected": "UNANSWERABLE"
    }
]


semantic_service = SemanticSearchService()
bm25_service = BM25SearchService()

print("\nRETRIEVAL RELEVANCE BASELINE")
print("=" * 80)

for case in TEST_CASES:

    question = case["question"]

    semantic_results = semantic_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=1
    )

    bm25_results = bm25_service.search(
        document_id=DOCUMENT_ID,
        query=question,
        top_k=1
    )

    semantic_score = (
        semantic_results[0]["similarity"]
        if semantic_results else 0.0
    )

    bm25_score = (
        bm25_results[0]["bm25_score"]
        if bm25_results else 0.0
    )

    semantic_chunk = (
        semantic_results[0]["chunk_index"]
        if semantic_results else None
    )

    bm25_chunk = (
        bm25_results[0]["chunk_index"]
        if bm25_results else None
    )

    print(f"\nQuestion: {question}")
    print(f"Expected: {case['expected']}")
    print(f"Semantic score: {semantic_score:.4f}")
    print(f"Semantic chunk: {semantic_chunk}")
    print(f"BM25 score: {bm25_score:.4f}")
    print(f"BM25 chunk: {bm25_chunk}")

print("\nBaseline evaluation completed.")