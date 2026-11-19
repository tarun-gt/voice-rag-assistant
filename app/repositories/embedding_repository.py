
import json
from pathlib import Path


class EmbeddingRepository:

    def __init__(self):
        self.embedding_directory = Path("storage/embeddings")
        self.embedding_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        document_id: str,
        embedded_chunks: list[dict]
    ) -> list[dict]:

        # Associate every embedded chunk with its parent audio document
        document_embedded_chunks = []

        for chunk in embedded_chunks:
            document_embedded_chunk = {
                **chunk,
                "document_id": document_id
            }

            document_embedded_chunks.append(document_embedded_chunk)

        # Create a separate JSON file for this document
        embedding_path = (
            self.embedding_directory / f"{document_id}.json"
        )

        with open(embedding_path, "w", encoding="utf-8") as file:
            json.dump(
                document_embedded_chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        return document_embedded_chunks

    def find_by_document_id(
        self,
        document_id: str
    ) -> list[dict] | None:

        embedding_path = (
            self.embedding_directory / f"{document_id}.json"
        )

        if not embedding_path.exists():
            return None

        with open(embedding_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def delete(self, document_id: str) -> bool:

        embedding_path = (
            self.embedding_directory / f"{document_id}.json"
        )

        if not embedding_path.exists():
            return False

        embedding_path.unlink()
        return True
