from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import ollama
import uvicorn

OLLAMA_URL = "http://localhost:11434"  # Ollama 內建 REST
LLM_MODEL = "qwen2.5:7b-instruct"  # 預設模型
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"

class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str
    stream: bool = False

app = FastAPI(title="Local LLM via Ollama")

@app.post("/chat")
def chat(req: ChatRequest):
    merged_system = DEFAULT_SYSTEM_PROMPT
    if req.system != DEFAULT_SYSTEM_PROMPT:  # 表示用戶有自訂 system
        merged_system = f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system}"
    messages = [
        {"role": "system", "content": merged_system},
        {"role": "user", "content": req.user}
    ]
    resp = ollama.chat(
      model=req.model,
      messages=messages,
      stream=False
    )
    content = resp.get("message").get("content")
    return {"answer": content}

if __name__ == "__main__":
    uvicorn.run("main:app", port= 5000, reload=True)
