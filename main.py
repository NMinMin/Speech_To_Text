from fastapi import FastAPI, UploadFile, File, HTTPException
import httpx
import os

app = FastAPI()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Chưa cấu hình HF_API_TOKEN trong Environment!")

    audio_data = await file.read()
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": file.content_type or "audio/mpeg"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(API_URL, headers=headers, data=audio_data)
        
        # Nếu Hugging Face trả mã lỗi (4xx, 5xx)
        if response.status_code != 200:
            return {
                "error_from_hf": True,
                "status_code": response.status_code,
                "detail": response.text
            }

        result = response.json()
        return {"text": result.get("text", "")}

    except Exception as e:
        # Bắt toàn bộ lỗi kết nối/timeout
        return {"error_gateway": True, "message": str(e)}