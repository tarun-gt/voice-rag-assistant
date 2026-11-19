
from app.services.embedding_service import EmbeddingService


class ChunkEmbeddingService:

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def generate_embeddings(
        self,
        chunks: list[dict]
    ) -> list[dict]:

        embedded_chunks = []

        for chunk in chunks:

            # Generate an embedding from the chunk's text
            embedding = self.embedding_service.generate_embedding(
                chunk["text"]
            )

            # Preserve existing chunk data and attach its embedding
            embedded_chunk = {
                **chunk,
                "embedding": embedding
            }

            embedded_chunks.append(embedded_chunk)

        return embedded_chunks
