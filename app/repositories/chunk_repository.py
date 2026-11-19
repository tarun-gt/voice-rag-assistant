
import json
from pathlib import Path


class ChunkRepository:

    def __init__(self):
        self.chunk_directory = Path("storage/chunks")
        self.chunk_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, document_id: str, chunks: list[dict]) -> list[dict]:

        # Associate every chunk with its parent audio document
        document_chunks = []

        for chunk in chunks:
            document_chunk = {
                "document_id": document_id,
                **chunk
            }

            document_chunks.append(document_chunk)

        # Create a separate JSON file for this document
        chunk_path = self.chunk_directory / f"{document_id}.json"

        with open(chunk_path, "w", encoding="utf-8") as file:
            json.dump(
                document_chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        return document_chunks

    def find_by_document_id(self, document_id: str) -> list[dict] | None:

        chunk_path = self.chunk_directory / f"{document_id}.json"

        if not chunk_path.exists():
            return None

        with open(chunk_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def delete(self, document_id: str) -> bool:

        chunk_path = self.chunk_directory / f"{document_id}.json"

        if not chunk_path.exists():
            return False

        chunk_path.unlink()
        return True
