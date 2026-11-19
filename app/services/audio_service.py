import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException

from app.repositories.audio_repository import AudioRepository
from app.services.transcript_processing_service import (
    TranscriptProcessingService
)

class AudioService:

    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

    def __init__(self):
        self.repository = AudioRepository()
        self.audio_directory = Path("storage/audio")
        self.audio_directory.mkdir(parents=True, exist_ok=True)
        self.transcript_processing_service = None

    async def upload_audio(self, file: UploadFile) -> dict:

        # 1. Validate filename and extension
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is missing"
            )

        extension = Path(file.filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format. Allowed formats: MP3, WAV, M4A"
            )

        # 2. Generate a unique document ID
        document_id = str(uuid.uuid4())

        # 3. Create a safe server-side filename
        audio_path = self.audio_directory / f"{document_id}{extension}"

        # 4. Save the uploaded file
        file_size = 0

        try:
            with open(audio_path, "wb") as audio_file:
                while chunk := await file.read(1024 * 1024):
                    audio_file.write(chunk)
                    file_size += len(chunk)

        except Exception:
            audio_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to save audio file"
            )

        # 5. Prepare metadata
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": extension.lstrip("."),
            "file_size": file_size,
            "status": "uploaded",
            "message": "Audio uploaded successfully"
        }

        # 6. Persist initial metadata
        try:
            self.repository.save(metadata)
        except Exception:
            audio_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to save audio metadata"
            )

        # 7. Update status to transcribing
        metadata["status"] = "transcribing"
        metadata["message"] = "Audio uploaded. Transcription in progress."

        self.repository.save(metadata)

        # 8. Initialize transcription service only when needed
        if self.transcript_processing_service is None:
            self.transcript_processing_service = TranscriptProcessingService()

        # 9. Transcribe audio and save transcript
        try:
            self.transcript_processing_service.process_audio(
                document_id=document_id,
                audio_path=str(audio_path)
            )

            metadata["status"] = "transcribed"
            metadata["message"] = "Audio uploaded and transcribed successfully."

        except Exception:
            metadata["status"] = "transcription_failed"
            metadata["message"] = (
                "Audio uploaded successfully, but transcription failed."
            )

        # 10. Persist final processing status
        self.repository.save(metadata)

        return metadata

    
    def get_audio(self, document_id: str) -> dict:
        metadata = self.repository.find_by_id(document_id)

        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Audio document '{document_id}' not found"
            )

        return metadata

    
    def delete_audio(self, document_id: str) -> dict:
        # 1. Retrieve document metadata
        metadata = self.repository.find_by_id(document_id)

        if metadata is None:
            raise HTTPException(
                status_code=404,
                detail=f"Audio document '{document_id}' not found"
            )

        # 2. Identify the original audio file
        extension = metadata["file_type"]
        audio_path = self.audio_directory / f"{document_id}.{extension}"

        # 3. Delete the physical audio file
        if audio_path.exists():
            audio_path.unlink()

        # 4. Delete metadata
        self.repository.delete(document_id)

        return {
            "document_id": document_id,
            "message": "Audio deleted successfully"
        }
