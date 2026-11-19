
from fastapi import FastAPI

from app.api.v1.audio import router as audio_router

app = FastAPI(
    title="Audio RAG Assistant",
    description="Ask questions about uploaded audio recordings using RAG.",
    version="1.0.0"
)


app.include_router(
    audio_router,
    prefix="/api/v1/audio",
    tags=["Audio"]
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Audio RAG Assistant",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "UP",
        "application": "audio-rag-assistant"
    }
