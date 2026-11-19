from app.services.embedding_service import EmbeddingService
from app.repositories.embedding_repository import EmbeddingRepository
import numpy as np


class SemanticSearchService:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embedding_repository = EmbeddingRepository()

    def _generate_query_embedding(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        return self.embedding_service.generate_embedding(query)

    def _calculate_similarity(
        self,
        query_embedding: list[float],
        chunk_embedding: list[float]
    ) -> float:

        similarity = np.dot(
            query_embedding,
            chunk_embedding
        )

        return float(similarity)

    def _get_document_embeddings(
        self,
        document_id: str
    ) -> list[dict]:

        embedded_chunks = self.embedding_repository.find_by_document_id(
            document_id
        )

        if embedded_chunks is None:
            raise ValueError(
                f"No embeddings found for document: {document_id}"
            )

        return embedded_chunks

    def _rank_chunks(
        self,
        query_embedding: list[float],
        embedded_chunks: list[dict]
    ) -> list[dict]:

        ranked_chunks = []

        for chunk in embedded_chunks:

            similarity = self._calculate_similarity(
                query_embedding,
                chunk["embedding"]
            )

            ranked_chunk = {
                **chunk,
                "similarity": similarity
            }

            ranked_chunks.append(ranked_chunk)

        # Sort chunks by similarity in descending order
        ranked_chunks.sort(
            key=lambda chunk: chunk["similarity"],
            reverse=True
        )

        return ranked_chunks

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # 1. Generate embedding for the user's question
        query_embedding = self._generate_query_embedding(query)

        # 2. Retrieve embedded chunks for the requested document
        embedded_chunks = self._get_document_embeddings(document_id)

        # 3. Calculate similarity and rank all chunks
        ranked_chunks = self._rank_chunks(
            query_embedding,
            embedded_chunks
        )

        # 4. Return only the top K relevant chunks
        return ranked_chunks[:top_k]