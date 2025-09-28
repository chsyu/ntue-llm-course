# main.py
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from texts import texts  # 你的教學語料（List[str]）

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# =====================
# 向量庫設定
# =====================
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_store"   # Chroma 0.4+ 自動持久化
EMBED_MODEL = "nomic-embed-text"          # 嵌入模型
LLM_MODEL   = "qwen2.5:7b-instruct"       # 回答用 LLM
DEFAULT_SYSTEM_PROMPT = (
    "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"
    "若無法從提供的內容得到答案，請直說不知道。"
)

def build_or_load_vectorstore(embeddings: OllamaEmbeddings, seed_texts: List[str]) -> Chroma:
    if PERSIST_DIR.exists():
        print("偵測到既有向量庫，從磁碟載入…")
        return Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )
    print("未偵測到向量庫，建立新索引（Chroma 0.4+ 自動持久化）…")
    return Chroma.from_texts(
        texts=seed_texts,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

# =====================
# 啟動時初始化：Embeddings / VectorStore / Retriever
# =====================
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vs = build_or_load_vectorstore(embeddings, texts)

# 使用 MMR 的 Retriever（更穩定的多樣性）
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.2},
)

# 將檢索結果組成文字餵給 LLM
def format_docs(docs: List) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] {d.page_content}")
    return "\n".join(parts)

# =====================
# FastAPI 介面
# =====================
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

app = FastAPI(title="RAG over Chroma + Ollama (prompt | llm)")

@app.post("/chat")
def chat(req: ChatRequest):
    # 合併 system 提示
    sys_merged = (
        DEFAULT_SYSTEM_PROMPT
        if req.system == DEFAULT_SYSTEM_PROMPT
        else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"
    )

    # 先做檢索（RAG）
    retrieved = retriever.get_relevant_documents(req.user)
    context = format_docs(retrieved)

    # 提示模板：把 context 明確塞進去，要求模型僅根據 context 回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("user",
         "你將根據以下提供的內容回答問題。若內容不足以回答，請說你不知道。\n\n"
         "【內容】\n{context}\n\n"
         "【問題】\n{question}")
    ])

    # 建立 LLM（每次請求可用不同模型）
    llm = ChatOllama(model=req.model, temperature=0.2)

    # LCEL：prompt -> llm（檢索在 chain 外）
    chain = prompt | llm

    result = chain.invoke({"system": sys_merged, "context": context, "question": req.user})
    answer = result.content

    # 回傳來源片段（教學用途）
    sources = [d.page_content for d in retrieved]

    return {
        "answer": answer,
        "sources": sources,
        "k": 4,
        "mmr": {"fetch_k": 20, "lambda_mult": 0.2}
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)