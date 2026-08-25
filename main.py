from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
import os

app = FastAPI()

# Lấy Hugging Face API Token từ biến môi trường
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

@app.get("/")
def root():
    return {"message": "Speech to Text API Gateway Ready"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Thiếu HF_API_TOKEN trong biến môi trường!")

    # Đọc dữ liệu file âm thanh
    audio_bytes = await file.read()
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": file.content_type or "audio/wav"
    }

    # Gửi request sang Hugging Face Serverless API
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(API_URL, headers=headers, data=audio_bytes)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()
    return {"text": result.get("text", "")}