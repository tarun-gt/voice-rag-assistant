
from app.services.embedding_service import EmbeddingService


# 1. Initialize embedding service
embedding_service = EmbeddingService()


# 2. Generate an embedding
text = "Artificial intelligence is transforming the world."

embedding = embedding_service.generate_embedding(text)


# 3. Validate embedding
assert isinstance(embedding, list)
assert len(embedding) == 384
assert all(isinstance(value, float) for value in embedding)

print("\nEmbedding generated successfully!")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First five values: {embedding[:5]}")


# 4. Validate normalization
magnitude = sum(value ** 2 for value in embedding) ** 0.5

print(f"Embedding magnitude: {magnitude}")

assert abs(magnitude - 1.0) < 0.0001

print("Embedding normalization test passed!")


# 5. Validate empty input
try:
    embedding_service.generate_embedding("   ")
    assert False, "Expected ValueError for empty input"
except ValueError:
    print("Empty input validation passed!")


print("\nAll embedding service tests passed!")
