
from app.services.hybrid_search_service import HybridSearchService
from app.services.cross_encoder_reranking_service import (
    CrossEncoderRerankingService
)


DOCUMENT_ID = "e947268a-8e5d-4664-bd89-e1e9edaa78f7"

QUERY = "Which university does the speaker say they attended?"


print("Initializing hybrid search service...")

hybrid_service = HybridSearchService()

print("\nRetrieving candidate chunks...")

candidate_chunks = hybrid_service.search(
    document_id=DOCUMENT_ID,
    query=QUERY,
    top_k=10
)

print(f"Retrieved {len(candidate_chunks)} candidate chunks.")

print("\nInitializing reranking service...")

reranking_service = CrossEncoderRerankingService()

print("\nReranking candidate chunks...")

results = reranking_service.rerank(
    query=QUERY,
    candidate_chunks=candidate_chunks,
    top_k=5
)

print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for rank, chunk in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Chunk: {chunk['chunk_index']}")
    print(f"Reranker score: {chunk['reranker_score']:.4f}")
    print(f"RRF score: {chunk['rrf_score']:.6f}")
    print(f"Text: {chunk['text'][:250]}")

print("\nReranking validation...")

assert len(results) <= 5
assert all("reranker_score" in chunk for chunk in results)
assert results[0]["reranker_score"] >= results[-1]["reranker_score"]

print("Cross-encoder reranking tests passed!")
