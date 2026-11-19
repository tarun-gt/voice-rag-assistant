
from pydantic import BaseModel


class AudioUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    message: str
