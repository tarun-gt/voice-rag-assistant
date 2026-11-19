
from app.services.ranking_fusion_service import RankingFusionService


# Sample semantic search results
semantic_results = [
    {"chunk_index": 16, "text": "Chunk 16"},
    {"chunk_index": 39, "text": "Chunk 39"},
    {"chunk_index": 0, "text": "Chunk 0"}
]


# Sample BM25 search results
bm25_results = [
    {"chunk_index": 0, "text": "Chunk 0"},
    {"chunk_index": 1, "text": "Chunk 1"},
    {"chunk_index": 36, "text": "Chunk 36"}
]


print("Initializing ranking fusion service...")

fusion_service = RankingFusionService(rrf_constant=60)


print("\nStarting ranking fusion...")

results = fusion_service.fuse(
    semantic_results=semantic_results,
    bm25_results=bm25_results,
    top_k=5
)


# 1. Verify the number of unique chunks
assert len(results) == 5

print("Unique chunk retrieval validation passed!")


# 2. Verify Chunk 0 receives contributions from both searches
chunk_0 = next(
    chunk for chunk in results
    if chunk["chunk_index"] == 0
)

expected_chunk_0_score = (
    1 / (60 + 3) +
    1 / (60 + 1)
)

assert abs(
    chunk_0["rrf_score"] - expected_chunk_0_score
) < 1e-10

print("Combined score calculation validation passed!")


# 3. Verify ranking order
rrf_scores = [
    chunk["rrf_score"]
    for chunk in results
]

assert rrf_scores == sorted(
    rrf_scores,
    reverse=True
)

print("RRF ranking validation passed!")


# 4. Verify Chunk 0 ranks first
assert results[0]["chunk_index"] == 0

print("Combined ranking validation passed!")


# 5. Display results
print("\nFused search results:")

for rank, chunk in enumerate(results, start=1):

    print("\n" + "-" * 50)

    print(f"Rank: {rank}")
    print(f"Chunk index: {chunk['chunk_index']}")
    print(f"RRF score: {chunk['rrf_score']:.6f}")


print("\nAll ranking fusion tests passed!")
