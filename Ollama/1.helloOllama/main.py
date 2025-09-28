from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uvicorn

OLLAMA_URL = "http://localhost:11434"  # Ollama 內建 REST

class GenerateRequest(BaseModel):
    model: str = "qwen2.5:7b-instruct"
    prompt: str  
    stream: bool = False

app = FastAPI(title="Local LLM via Ollama")

@app.post("/generate")
def generate(req: GenerateRequest):
    payload = {
        "model": req.model,
        "prompt": req.prompt,
        "stream": req.stream
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    content = data.get("response", "")
    return {"answer": content}

if __name__ == "__main__":
    uvicorn.run("main:app", port= 5000, reload=True)
