from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

@app.get("/")
def root():
    return {"status": "Speech-to-Text Gateway is running"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_TOKEN:
        return {
            "error": True,
            "message": "Thiếu HF_API_TOKEN trong Environment Variables trên Render!"
        }

    try:
        audio_data = await file.read()
        
        # Nhận diện Content-Type tự động (hỗ trợ cả audio/mpeg cho file mp3)
        content_type = file.content_type or "audio/mpeg"
        
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN.strip()}",
            "Content-Type": content_type
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(API_URL, headers=headers, data=audio_data)

        if response.status_code != 200:
            return {
                "error_from_hf": True,
                "status_code": response.status_code,
                "detail": response.text
            }

        result = response.json()
        return {"text": result.get("text", "")}

    except Exception as e:
        return {
            "error_gateway": True,
            "message": str(e)
        }