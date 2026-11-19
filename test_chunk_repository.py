
from pathlib import Path

from app.repositories.chunk_repository import ChunkRepository


repository = ChunkRepository()

document_id = "test-document-123"

sample_chunks = [
    {
        "chunk_index": 0,
        "text": "Artificial intelligence is transforming the world.",
        "start": 0.0,
        "end": 5.0
    },
    {
        "chunk_index": 1,
        "text": "Machine learning helps computers learn from data.",
        "start": 5.0,
        "end": 10.0
    }
]


# 1. Test saving chunks
saved_chunks = repository.save(document_id, sample_chunks)

assert len(saved_chunks) == 2

assert all(
    chunk["document_id"] == document_id
    for chunk in saved_chunks
)

print("Chunk save test passed!")


# 2. Verify physical JSON file creation
chunk_file = Path(
    f"storage/chunks/{document_id}.json"
)

assert chunk_file.exists()

print("Chunk file creation test passed!")


# 3. Test retrieving chunks
retrieved_chunks = repository.find_by_document_id(document_id)

assert retrieved_chunks == saved_chunks

print("Chunk retrieval test passed!")


# 4. Test retrieving a nonexistent document
missing_chunks = repository.find_by_document_id("unknown-document")

assert missing_chunks is None

print("Missing document test passed!")


# 5. Test deleting chunks
deleted = repository.delete(document_id)

assert deleted is True
assert not chunk_file.exists()

print("Chunk deletion test passed!")


# 6. Test deleting an already deleted document
deleted_again = repository.delete(document_id)

assert deleted_again is False

print("Repeated deletion test passed!")


print("\nAll chunk repository tests passed!")
