
from app.services.chunk_embedding_service import ChunkEmbeddingService


# 1. Initialize the service
service = ChunkEmbeddingService()


# 2. Prepare sample chunks
chunks = [
    {
        "document_id": "test-document-001",
        "chunk_index": 0,
        "text": "Artificial intelligence is transforming the world.",
        "start": 0.0,
        "end": 10.0
    },
    {
        "document_id": "test-document-001",
        "chunk_index": 1,
        "text": "Machine learning helps computers learn from data.",
        "start": 10.0,
        "end": 20.0
    },
    {
        "document_id": "test-document-001",
        "chunk_index": 2,
        "text": "Deep learning uses neural networks to process information.",
        "start": 20.0,
        "end": 30.0
    }
]


# 3. Generate embeddings
embedded_chunks = service.generate_embeddings(chunks)


# 4. Validate the number of chunks
assert len(embedded_chunks) == len(chunks)

print("Chunk count validation passed!")


# 5. Validate each embedded chunk
for chunk in embedded_chunks:

    assert "embedding" in chunk
    assert len(chunk["embedding"]) == 384

    assert chunk["document_id"] == "test-document-001"
    assert chunk["text"]
    assert chunk["start"] <= chunk["end"]

print("Embedding dimension validation passed!")
print("Original chunk data preservation passed!")


# 6. Validate embedding normalization
for chunk in embedded_chunks:

    magnitude = sum(
        value ** 2
        for value in chunk["embedding"]
    ) ** 0.5

    assert abs(magnitude - 1.0) < 0.0001

print("Embedding normalization validation passed!")


# 7. Display results
print("\nGenerated embedded chunks:")

for chunk in embedded_chunks:

    print(
        f"Chunk {chunk['chunk_index']}: "
        f"{len(chunk['embedding'])} dimensions, "
        f"{chunk['start']}s - {chunk['end']}s"
    )


print("\nAll chunk embedding service tests passed!")
