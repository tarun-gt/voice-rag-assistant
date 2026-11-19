
from pathlib import Path
import json
import uuid

from app.services.transcript_processing_service import (
    TranscriptProcessingService
)
from app.repositories.transcript_repository import TranscriptRepository


# 1. Generate a unique document ID
document_id = str(uuid.uuid4())

# 2. Define the audio file
audio_path = Path("storage/audio/sample.mp3")

if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")


# 3. Initialize processing service
print("Initializing transcript processing service...")

processing_service = TranscriptProcessingService()


# 4. Transcribe and persist the transcript
print("\nStarting end-to-end processing...")

transcript = processing_service.process_audio(
    document_id=document_id,
    audio_path=str(audio_path)
)


# 5. Display processing results
print("\nProcessing completed!")

print(f"Document ID: {document_id}")
print(f"Audio filename: {transcript['audio_filename']}")
print(f"Language: {transcript['language']}")
print(f"Duration: {transcript['duration']} seconds")
print(f"Total segments: {len(transcript['segments'])}")


# 6. Verify the transcript JSON file exists
transcript_path = (
    Path("storage/transcripts") / f"{document_id}.json"
)

print("\nTranscript file exists:", transcript_path.exists())


# 7. Retrieve transcript using repository
repository = TranscriptRepository()

retrieved_transcript = repository.find_by_id(document_id)

print(
    "Transcript retrieved successfully:",
    retrieved_transcript is not None
)


# 8. Verify retrieved transcript content
if retrieved_transcript:
    print(
        "Retrieved segment count:",
        len(retrieved_transcript["segments"])
    )

    print("\nFirst three transcript segments:")

    for segment in retrieved_transcript["segments"][:3]:
        print(
            f"[{segment['start']}s - {segment['end']}s] "
            f"{segment['text']}"
        )

    assert retrieved_transcript == transcript
    print("\nTranscript content verification: PASSED")

else:
    raise AssertionError("Transcript could not be retrieved!")


print("\nEnd-to-end test completed successfully!")



from unittest.mock import patch

from app.services.transcript_processing_service import (
    TranscriptProcessingService
)
from app.repositories.transcript_repository import TranscriptRepository
from app.repositories.chunk_repository import ChunkRepository


AUDIO_PATH = "storage/audio/test-audio.mp3"


# Load the existing transcript to use as mocked transcription output
transcript_repository = TranscriptRepository()
existing_transcript = transcript_repository.find_by_id(document_id)

assert existing_transcript is not None, "Existing transcript not found!"


# Replace Whisper's service during construction so no model is loaded
with patch(
    "app.services.transcript_processing_service.TranscriptionService"
) as mock_transcription:

    mock_transcription.return_value.transcribe_audio.return_value = (
        existing_transcript
    )

    processing_service = TranscriptProcessingService()

    result = processing_service.process_audio(
        document_id=document_id,
        audio_path=AUDIO_PATH
    )


# Verify the transcript was returned
assert result == existing_transcript
print("Transcript processing passed!")


# Verify chunks were persisted by process_audio()
chunk_repository = ChunkRepository()
saved_chunks = chunk_repository.find_by_document_id(document_id)

assert saved_chunks is not None
assert len(saved_chunks) > 0

print(f"Chunk generation and persistence passed: {len(saved_chunks)} chunks")

print("\nAll transcript processing integration tests passed!")
