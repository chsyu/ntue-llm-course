from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import uvicorn

OLLAMA_URL = "http://localhost:11434"  # Ollama 內建 REST

class GenerateRequest(BaseModel):
    model: str = "qwen2.5:7b-instruct"
    prompt: str  
    stream: bool = False

app = FastAPI(title="Local LLM via Ollama")

@app.post("/generate")
def generate(req: GenerateRequest):
        resp = ollama.generate(
            model=req.model,
            prompt=req.prompt,
            stream=False
        )
        content = resp.get("response", "")
        return {"answer": content}

if __name__ == "__main__":
    uvicorn.run("main:app", port= 5000, reload=True)
