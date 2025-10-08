from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"

class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

app = FastAPI(title="LC + Ollama: chat")

@app.post("/chat")
async def chat(req: ChatRequest):
    sys_merged = DEFAULT_SYSTEM_PROMPT if req.system == DEFAULT_SYSTEM_PROMPT \
                 else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user", "{question}")
    ])

    llm = ChatOllama(model=req.model, temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    # 非同步呼叫 LLM
    result = await chain.ainvoke({"question": req.user})
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)