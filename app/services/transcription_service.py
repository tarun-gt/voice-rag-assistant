
from pathlib import Path

from faster_whisper import WhisperModel


class TranscriptionService:

    def __init__(self):
        print("Loading Faster-Whisper model...")

        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("Faster-Whisper model loaded successfully!")

    def transcribe_audio(self, audio_path: str) -> dict:

        # 1. Validate the audio file
        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}"
            )

        # 2. Perform transcription
        segments, info = self.model.transcribe(
            str(audio_file),
            beam_size=5
        )

        # 3. Collect timestamped segments
        transcript_segments = []

        for segment in segments:
            transcript_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        # 4. Return structured transcript data
        return {
            "audio_filename": audio_file.name,
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": transcript_segments
        }
