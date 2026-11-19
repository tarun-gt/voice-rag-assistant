
from app.repositories.embedding_repository import EmbeddingRepository


# 1. Initialize repository
repository = EmbeddingRepository()

document_id = "test-document-001"


# 2. Prepare sample embedded chunks
embedded_chunks = [
    {
        "document_id": document_id,
        "chunk_index": 0,
        "text": "Artificial intelligence is transforming the world.",
        "start": 0.0,
        "end": 10.0,
        "embedding": [0.1, 0.2, 0.3]
    },
    {
        "document_id": document_id,
        "chunk_index": 1,
        "text": "Machine learning helps computers learn from data.",
        "start": 10.0,
        "end": 20.0,
        "embedding": [0.4, 0.5, 0.6]
    }
]


# 3. Test saving embeddings
saved_chunks = repository.save(
    document_id,
    embedded_chunks
)

assert saved_chunks == embedded_chunks

print("Embedding save test passed!")


# 4. Verify physical JSON file creation
from pathlib import Path

embedding_file = (
    Path("storage/embeddings") / f"{document_id}.json"
)

assert embedding_file.exists()

print("Embedding file creation test passed!")


# 5. Test retrieving embeddings
retrieved_chunks = repository.find_by_document_id(document_id)

assert retrieved_chunks == embedded_chunks

print("Embedding retrieval test passed!")


# 6. Test retrieving a nonexistent document
missing_chunks = repository.find_by_document_id(
    "nonexistent-document"
)

assert missing_chunks is None

print("Missing document test passed!")


# 7. Test deleting embeddings
deleted = repository.delete(document_id)

assert deleted is True
assert not embedding_file.exists()

print("Embedding deletion test passed!")


# 8. Test repeated deletion
deleted_again = repository.delete(document_id)

assert deleted_again is False

print("Repeated deletion test passed!")


print("\nAll embedding repository tests passed!")
