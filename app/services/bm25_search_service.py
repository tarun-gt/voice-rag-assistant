
import re

from rank_bm25 import BM25Okapi

from app.repositories.chunk_repository import ChunkRepository


class BM25SearchService:

    def __init__(self):
        self.chunk_repository = ChunkRepository()

    def _tokenize(self, text: str) -> list[str]:

        if not text or not text.strip():
            return []

        # Convert text to lowercase and extract individual words
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # 1. Retrieve original chunks for the document
        chunks = self.chunk_repository.find_by_document_id(
            document_id
        )

        if chunks is None:
            raise ValueError(
                f"No chunks found for document: {document_id}"
            )

        if not chunks:
            return []

        # 2. Tokenize each chunk to build the BM25 corpus
        tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        # 3. Create the BM25 index
        bm25 = BM25Okapi(tokenized_chunks)

        # 4. Tokenize the user's query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        # 5. Calculate BM25 scores for all chunks
        scores = bm25.get_scores(tokenized_query)

        # 6. Attach scores to their corresponding chunks
        ranked_chunks = []

        for chunk, score in zip(chunks, scores):

            ranked_chunk = {
                **chunk,
                "bm25_score": float(score)
            }

            ranked_chunks.append(ranked_chunk)

        # 7. Sort by BM25 score in descending order
        ranked_chunks.sort(
            key=lambda chunk: chunk["bm25_score"],
            reverse=True
        )

        # 8. Return only the top K results
        return ranked_chunks[:top_k]
