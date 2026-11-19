
from sentence_transformers import CrossEncoder


class CrossEncoderRerankingService:

    def __init__(self):
        print("Loading cross-encoder model...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )

        print("Cross-encoder model loaded successfully!")

    def rerank(
        self,
        query: str,
        candidate_chunks: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not candidate_chunks:
            return []

        pairs = [
            [query, chunk["text"]]
            for chunk in candidate_chunks
        ]

        scores = self.model.predict(pairs)

        reranked_chunks = []

        for chunk, score in zip(candidate_chunks, scores):
            reranked_chunk = {
                **chunk,
                "reranker_score": float(score)
            }

            reranked_chunks.append(reranked_chunk)

        reranked_chunks.sort(
            key=lambda chunk: chunk["reranker_score"],
            reverse=True
        )

        return reranked_chunks[:top_k]
