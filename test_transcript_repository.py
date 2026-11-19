
from app.repositories.transcript_repository import TranscriptRepository


# 1. Initialize repository
repository = TranscriptRepository()

# 2. Prepare sample transcript
document_id = "test-document-001"

transcript = {
    "audio_filename": "sample.mp3",
    "language": "en",
    "duration": 120.5,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hello, welcome to the session."
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "Today we are discussing artificial intelligence."
        }
    ]
}

# 3. Save transcript
saved_transcript = repository.save(document_id, transcript)
print("Transcript saved:", saved_transcript is not None)

# 4. Retrieve transcript
retrieved_transcript = repository.find_by_id(document_id)
print("Transcript retrieved:", retrieved_transcript is not None)

print("Retrieved filename:", retrieved_transcript["audio_filename"])
print("Retrieved segments:", len(retrieved_transcript["segments"]))

# 5. Delete transcript
deleted = repository.delete(document_id)
print("Transcript deleted:", deleted)

# 6. Verify deletion
deleted_transcript = repository.find_by_id(document_id)
print("Transcript exists after deletion:", deleted_transcript is not None)
