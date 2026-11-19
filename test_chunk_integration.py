
from app.repositories.transcript_repository import TranscriptRepository
from app.repositories.chunk_repository import ChunkRepository
from app.services.text_chunking_service import TextChunkingService


DOCUMENT_ID = "5780fcc3-57c8-4705-8cf1-4662061561a5"


# 1. Load the existing transcript
transcript_repository = TranscriptRepository()
transcript = transcript_repository.find_by_id(DOCUMENT_ID)

assert transcript is not None, "Transcript not found!"

print("Transcript retrieval passed!")


# 2. Generate chunks
chunking_service = TextChunkingService()
chunks = chunking_service.chunk_transcript(transcript)

assert len(chunks) > 0, "No chunks generated!"

print(f"Chunk generation passed! Generated {len(chunks)} chunks.")


# 3. Persist chunks
chunk_repository = ChunkRepository()
saved_chunks = chunk_repository.save(DOCUMENT_ID, chunks)

assert len(saved_chunks) == len(chunks)
assert all(
    chunk["document_id"] == DOCUMENT_ID
    for chunk in saved_chunks
)

print("Chunk persistence passed!")


# 4. Retrieve persisted chunks
retrieved_chunks = chunk_repository.find_by_document_id(DOCUMENT_ID)

assert retrieved_chunks == saved_chunks

print("Chunk retrieval passed!")


# 5. Validate chunk properties
for index, chunk in enumerate(retrieved_chunks):
    assert chunk["chunk_index"] == index
    assert chunk["text"].strip()
    assert len(chunk["text"]) <= 500
    assert chunk["start"] <= chunk["end"]

print("Chunk validation passed!")

print("\nAll chunk integration tests passed!")
