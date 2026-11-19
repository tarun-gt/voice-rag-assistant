
import json
from pathlib import Path

from faster_whisper import WhisperModel


# 1. Define input and output paths
audio_path = Path("storage/audio/sample.mp3")
transcript_directory = Path("storage/transcripts")

transcript_directory.mkdir(parents=True, exist_ok=True)

if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")


# 2. Load the Whisper model
print("Loading Faster-Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")


# 3. Transcribe the audio
print("Starting transcription...")

segments, info = model.transcribe(
    str(audio_path),
    beam_size=5
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")
print(f"Audio duration: {info.duration:.2f} seconds")


# 4. Collect timestamped transcript segments
transcript_segments = []

for segment in segments:
    segment_data = {
        "start": round(segment.start, 2),
        "end": round(segment.end, 2),
        "text": segment.text.strip()
    }

    transcript_segments.append(segment_data)

    print(
        f"[{segment_data['start']:.2f}s - "
        f"{segment_data['end']:.2f}s] "
        f"{segment_data['text']}"
    )


# 5. Save the transcript as JSON
output_path = transcript_directory / f"{audio_path.stem}.json"

transcript_data = {
    "audio_filename": audio_path.name,
    "language": info.language,
    "duration": info.duration,
    "segments": transcript_segments
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        transcript_data,
        file,
        indent=4,
        ensure_ascii=False
    )

print(f"\nTranscription completed!")
print(f"Total segments: {len(transcript_segments)}")
print(f"Transcript saved at: {output_path}")
