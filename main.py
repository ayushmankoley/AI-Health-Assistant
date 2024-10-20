from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import json
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str
    language: str

class Response(BaseModel):
    answer: str

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-90b-text-preview",
            "messages": [
                {"role": "system", "content": "You are an advanced AI-powered virtual health assistant. Provide helpful medical advice, but always recommend consulting a real doctor for serious issues. Don't answer questions which are not related to health/medical advice"},
                {"role": "user", "content": f"Translate to {query.language} if necessary. Provide a medical consultation for: {query.text}"}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return Response(answer=result["choices"][0]["message"]["content"])
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI-Powered Virtual Health Assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)