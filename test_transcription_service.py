
from pathlib import Path

from app.services.transcription_service import TranscriptionService


# 1. Initialize the transcription service
transcription_service = TranscriptionService()


# 2. Define the audio file
audio_path = Path("storage/audio/sample.mp3")


# 3. Transcribe the audio
print("\nStarting transcription...")

transcript = transcription_service.transcribe_audio(
    str(audio_path)
)


# 4. Display transcript information
print("\nTranscription completed!")

print(f"Filename: {transcript['audio_filename']}")
print(f"Language: {transcript['language']}")
print(f"Duration: {transcript['duration']} seconds")
print(f"Total segments: {len(transcript['segments'])}")


# 5. Display the first five segments
print("\nFirst five transcript segments:")

for segment in transcript["segments"][:5]:
    print(
        f"[{segment['start']}s - {segment['end']}s] "
        f"{segment['text']}"
    )
