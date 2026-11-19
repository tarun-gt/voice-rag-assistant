
from app.services.semantic_search_service import SemanticSearchService
from app.services.bm25_search_service import BM25SearchService
from app.services.ranking_fusion_service import RankingFusionService
from app.services.cross_encoder_reranking_service import (
    CrossEncoderRerankingService
)


class HybridSearchService:

    def __init__(self):
        self.semantic_search_service = SemanticSearchService()
        self.bm25_search_service = BM25SearchService()
        self.ranking_fusion_service = RankingFusionService()
        self.cross_encoder_service = CrossEncoderRerankingService()

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = max(top_k * 2, 10)

        semantic_results = self.semantic_search_service.search(
            document_id=document_id,
            query=query,
            top_k=candidate_k
        )

        bm25_results = self.bm25_search_service.search(
            document_id=document_id,
            query=query,
            top_k=candidate_k
        )

        fused_results = self.ranking_fusion_service.fuse(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            top_k=candidate_k
        )

        reranked_results = self.cross_encoder_service.rerank(
            query=query,
            candidate_chunks=fused_results,
            top_k=top_k
        )

        return reranked_results
