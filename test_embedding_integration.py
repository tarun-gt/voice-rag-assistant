
from app.repositories.embedding_repository import EmbeddingRepository


document_id = "f310c6f7-41a2-4a8c-a238-c7bf238e5b19"

repository = EmbeddingRepository()


# 1. Retrieve embedded chunks
embedded_chunks = repository.find_by_document_id(document_id)

assert embedded_chunks is not None
assert len(embedded_chunks) > 0

print(f"Embedding retrieval passed: {len(embedded_chunks)} chunks")


# 2. Validate every embedded chunk
for chunk in embedded_chunks:

    assert chunk["document_id"] == document_id
    assert "chunk_index" in chunk
    assert "text" in chunk
    assert "start" in chunk
    assert "end" in chunk
    assert "embedding" in chunk

    assert len(chunk["embedding"]) == 384

print("Embedding structure validation passed!")


# 3. Validate chunk indexes
indexes = [
    chunk["chunk_index"]
    for chunk in embedded_chunks
]

assert indexes == list(range(len(embedded_chunks)))

print("Chunk index validation passed!")


# 4. Validate embedding normalization
for chunk in embedded_chunks:

    magnitude = sum(
        value ** 2
        for value in chunk["embedding"]
    ) ** 0.5

    assert abs(magnitude - 1.0) < 0.0001

print("Embedding normalization validation passed!")


# 5. Display sample
first_chunk = embedded_chunks[0]

print("\nFirst embedded chunk:")
print(f"Text: {first_chunk['text']}")
print(f"Timestamp: {first_chunk['start']} - {first_chunk['end']}")
print(f"Embedding dimensions: {len(first_chunk['embedding'])}")
print(f"First five values: {first_chunk['embedding'][:5]}")


print("\nAll embedding integration tests passed!")
