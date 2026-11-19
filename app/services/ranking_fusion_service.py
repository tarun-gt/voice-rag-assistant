
class RankingFusionService:

    def __init__(self, rrf_constant: int = 60):

        if rrf_constant <= 0:
            raise ValueError("RRF constant must be greater than zero")

        self.rrf_constant = rrf_constant

    def fuse(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Store the combined scores using chunk_index as the key
        fused_results = {}

        # Process semantic search rankings
        for rank, chunk in enumerate(semantic_results, start=1):

            chunk_index = chunk["chunk_index"]

            if chunk_index not in fused_results:
                fused_results[chunk_index] = {
                    **chunk,
                    "rrf_score": 0.0
                }

            fused_results[chunk_index]["rrf_score"] += (
                1 / (self.rrf_constant + rank)
            )

        # Process BM25 search rankings
        for rank, chunk in enumerate(bm25_results, start=1):

            chunk_index = chunk["chunk_index"]

            if chunk_index not in fused_results:
                fused_results[chunk_index] = {
                    **chunk,
                    "rrf_score": 0.0
                }

            fused_results[chunk_index]["rrf_score"] += (
                1 / (self.rrf_constant + rank)
            )

        # Sort by combined RRF score in descending order
        ranked_results = sorted(
            fused_results.values(),
            key=lambda chunk: chunk["rrf_score"],
            reverse=True
        )

        return ranked_results[:top_k]
