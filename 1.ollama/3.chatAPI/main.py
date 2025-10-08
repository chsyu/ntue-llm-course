from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import uvicorn

OLLAMA_URL = "http://localhost:11434"  # Ollama 內建 REST
LLM_MODEL = "qwen2.5:7b-instruct"  # 預設模型
class Message(BaseModel):
    role: str
    content: str
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    messages: list[Message]
    stream: bool = False

class GenerateRequest(BaseModel):
    model: str = LLM_MODEL
    prompt: str  
    stream: bool = False

app = FastAPI(title="Local LLM via Ollama")

@app.post("/chat")
def chat(req: ChatRequest):
    resp = ollama.chat(
      model=req.model,
      messages=[msg.model_dump() for msg in req.messages], # Convert Pydantic models to dicts
      stream=False
    )
    content = resp.get("message").get("content")
    return {"answer": content}

@app.post("/generate")
def generate(req: GenerateRequest):
    resp = ollama.generate(
        model=req.model,
        prompt=req.prompt,
        stream=req.stream
    )
    content = resp.get("response", "")
    return {"answer": content}

if __name__ == "__main__":
    uvicorn.run("main:app", port= 5000, reload=True)
