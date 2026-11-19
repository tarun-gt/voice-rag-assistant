
import json
from pathlib import Path


class AudioRepository:

    def __init__(self):
        self.metadata_file = Path("storage/audio_metadata.json")

    def save(self, metadata: dict) -> dict:
        documents = self._load_all()

        document_id = metadata["document_id"]
        documents[document_id] = metadata

        with open(self.metadata_file, "w", encoding="utf-8") as file:
            json.dump(documents, file, indent=4)

        return metadata

    def find_by_id(self, document_id: str) -> dict | None:
        documents = self._load_all()
        return documents.get(document_id)

    def find_all(self) -> dict:
        return self._load_all()

    def delete(self, document_id: str) -> bool:
        documents = self._load_all()

        if document_id not in documents:
            return False

        del documents[document_id]

        with open(self.metadata_file, "w", encoding="utf-8") as file:
            json.dump(documents, file, indent=4)

        return True

    def _load_all(self) -> dict:
        if not self.metadata_file.exists():
            return {}

        with open(self.metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)
