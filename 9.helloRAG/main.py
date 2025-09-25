# main.py
from pathlib import Path
from typing import List, Optional
import re

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from texts import texts  # 你的教學語料

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# =====================
# 向量庫設定（沿用你的簡化版）
# =====================
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_store"
EMBED_MODEL = "nomic-embed-text"        # 嵌入模型
LLM_MODEL   = "qwen2.5:7b-instruct"     # 你原本預設的 LLM
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。若無法從提供的內容得到答案，請直說不知道。"

# （可選）清理可能造成編碼問題的奇怪碼點
_SURR = re.compile(r"[\ud800-\udfff]")
def clean(s: str) -> str:
    return _SURR.sub("", (s or "")).encode("utf-8", "ignore").decode("utf-8", "ignore")
def clean_list(xs: List[str]) -> List[str]:
    return [clean(x) for x in xs]

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
docs_texts = clean_list(texts)  # 保險：把語料做一次清理
vs = build_or_load_vectorstore(embeddings, docs_texts)

# 建立 Retriever（MMR 讓結果更穩定）
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.2},
)

# 將檢索結果組成文字餵給 LLM
def format_docs(docs: List) -> str:
    # 只取 page_content；如需顯示來源，可附 metadata
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

app = FastAPI(title="RAG over Chroma + Ollama")

@app.post("/chat")
def chat(req: ChatRequest):
    # 準備最終的 system prompt
    sys_merged = (
        DEFAULT_SYSTEM_PROMPT
        if req.system == DEFAULT_SYSTEM_PROMPT
        else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"
    )

    # 先做檢索（RAG）
    question = clean(req.user)
    retrieved = retriever.get_relevant_documents(question)
    context = format_docs(retrieved)

    # 提示模板：把 context 明確塞進去，要求模型「僅根據 context 回答」
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("user",
         "你將根據以下提供的內容回答問題。若內容不足以回答，請說你不知道。\n\n"
         "【內容】\n{context}\n\n"
         "【問題】\n{question}")
    ])

    # 建立 LLM（每次請求可用不同模型）
    llm = ChatOllama(model=req.model, temperature=0.2)

    # LCEL：prompt -> llm
    chain = prompt | llm

    result = chain.invoke({"system": sys_merged, "context": context, "question": question})
    answer = result.content

    # （可選）整理簡易來源，讓教學看得到「是哪些段落被用到」
    sources = [d.page_content for d in retrieved]

    return {
        "answer": answer,
        "sources": sources,     # 教學用途：讓學生看到被取用的片段
        "k": 4,                 # 本次檢索使用的 k
        "mmr": {"fetch_k": 20, "lambda_mult": 0.2}
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # 啟動：uvicorn main:app --reload --port 5000
    uvicorn.run("main:app", port=5000, reload=True)