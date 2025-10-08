from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 讀取 .env 檔案
load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # OpenAI 模型
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"

class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

app = FastAPI(title="LC + OpenAI: chat")

@app.post("/chat")
def chat(req: ChatRequest):
    sys_merged = DEFAULT_SYSTEM_PROMPT if req.system == DEFAULT_SYSTEM_PROMPT \
                 else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"

    # LangChain 提示模板（等價於 system + user）
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user", "{question}")
    ])

    # 改用 OpenAI
    llm = ChatOpenAI(
        model=req.model,
        temperature=0.3
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"question": req.user})
    return {"answer": result}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)