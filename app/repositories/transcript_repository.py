
import json
from pathlib import Path


class TranscriptRepository:

    def __init__(self):
        self.transcript_directory = Path("storage/transcripts")
        self.transcript_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, document_id: str, transcript: dict) -> dict:

        transcript_path = (
            self.transcript_directory / f"{document_id}.json"
        )

        with open(transcript_path, "w", encoding="utf-8") as file:
            json.dump(
                transcript,
                file,
                indent=4,
                ensure_ascii=False
            )

        return transcript

    def find_by_id(self, document_id: str) -> dict | None:

        transcript_path = (
            self.transcript_directory / f"{document_id}.json"
        )

        if not transcript_path.exists():
            return None

        with open(transcript_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def delete(self, document_id: str) -> bool:

        transcript_path = (
            self.transcript_directory / f"{document_id}.json"
        )

        if not transcript_path.exists():
            return False

        transcript_path.unlink()
        return True
