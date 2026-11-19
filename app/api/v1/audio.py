
from fastapi import APIRouter, UploadFile, File

from app.services.audio_service import AudioService
from app.schemas.audio import AudioUploadResponse


router = APIRouter()

audio_service = AudioService()


@router.post(
    "/upload",
    response_model=AudioUploadResponse
)
async def upload_audio(file: UploadFile = File(...)):
    return await audio_service.upload_audio(file)


@router.get(
    "/{document_id}",
    response_model=AudioUploadResponse
)
async def get_audio(document_id: str):
    return audio_service.get_audio(document_id)


@router.delete("/{document_id}")
async def delete_audio(document_id: str):
    return audio_service.delete_audio(document_id)
