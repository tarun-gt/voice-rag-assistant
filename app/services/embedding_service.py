
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        print("Embedding model loaded successfully!")

    def generate_embedding(self, text: str) -> list[float]:

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()
