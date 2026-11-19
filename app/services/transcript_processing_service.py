
from app.services.transcription_service import TranscriptionService
from app.repositories.transcript_repository import TranscriptRepository
from app.services.text_chunking_service import TextChunkingService
from app.repositories.chunk_repository import ChunkRepository
from app.services.chunk_embedding_service import ChunkEmbeddingService
from app.repositories.embedding_repository import EmbeddingRepository

class TranscriptProcessingService:

    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.transcript_repository = TranscriptRepository()
        self.text_chunking_service = TextChunkingService()
        self.chunk_repository = ChunkRepository()
        self.chunk_embedding_service = ChunkEmbeddingService()
        self.embedding_repository = EmbeddingRepository()

    
    def process_audio(
        self,
        document_id: str,
        audio_path: str
    ) -> dict:

        # 1. Generate transcript from audio
        transcript = self.transcription_service.transcribe_audio(
            audio_path
        )

        # 2. Persist transcript using the audio document ID
        saved_transcript = self.transcript_repository.save(
            document_id,
            transcript
        )

        # 3. Generate chunks from the saved transcript
        chunks = self.text_chunking_service.chunk_transcript(
            saved_transcript
        )

        # 4. Persist original chunks using the audio document ID
        self.chunk_repository.save(
            document_id,
            chunks
        )

        # 5. Generate embeddings for all chunks
        embedded_chunks = self.chunk_embedding_service.generate_embeddings(
            chunks
        )

        # 6. Persist embedded chunks using the audio document ID
        self.embedding_repository.save(
            document_id,
            embedded_chunks
        )

        # 7. Return the persisted transcript
        return saved_transcript

