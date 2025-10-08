from dotenv import load_dotenv
import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse  # ← 新增
from pydantic import BaseModel
import uvicorn

from texts import texts  # 你的教材語料

# LangChain + OpenAI + Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# =====================
# 基本設定
# =====================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBED_MODEL = "text-embedding-3-small"      # OpenAI embedding
LLM_MODEL   = "gpt-4o-mini"                 # OpenAI chat模型
INDEX_NAME  = "rag-demo-index"
DIMENSION   = 1536                          # text-embedding-3-small 對應 1536維
DEFAULT_SYSTEM_PROMPT = ""

# =====================
# 建立 / 載入 Pinecone
# =====================
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in {i["name"] for i in pc.list_indexes().get("indexes", [])}:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

# 如果索引為空 → 嵌入 texts → 上傳到 Pinecone
stats = index.describe_index_stats()
if stats.get("total_vector_count", 0) == 0:
    print("索引為空，上傳教材語料…")
    vectors = embeddings.embed_documents(texts)
    payload = [(f"id-{i}", v, {"text": t}) for i, (t, v) in enumerate(zip(texts, vectors))]
    index.upsert(vectors=payload)
    print("已完成上傳")

# 建立 VectorStore
vs = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

# 用 MMR 的 Retriever
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.2},
)

def format_docs(docs) -> str:
    return "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

# =====================
# Chain：檢索 → Prompt → LLM
# =====================
# ⚠️ 改用 retriever.invoke(...)，避免棄用警告
retriever_chain = RunnablePassthrough.assign(
    context=lambda x: format_docs(retriever.invoke(x["question"]))
)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一個精煉且忠實的助教。\n"
     "你的回答**必須完全依據提供的內容**，不得自行推測或引入外部知識。\n"
     "如果內容不足以回答，請直接回答：『我不知道』。"),
    ("user", "【內容】\n{context}\n\n【問題】\n{question}")
])

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
rag_chain = retriever_chain | prompt | llm

# =====================
# FastAPI
# =====================
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

class DeleteRequest(BaseModel):
    ids: List[str]

app = FastAPI(title="RAG (Pinecone + OpenAI)")

# 新增首頁 + 健康檢查，方便你直接打開瀏覽器測
@app.get("/")
def root():
    return {"ok": True, "service": "rag-api", "endpoints": ["/chat","/add_document","/add_documents","/delete_all_documents","/delete_by_ids","/list_stats","/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    print(f"[chat] received question: {req.user}")  # ← 簡單記錄請求
    sys_merged = (
        DEFAULT_SYSTEM_PROMPT
        if req.system == DEFAULT_SYSTEM_PROMPT
        else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"
    )
    result = rag_chain.invoke({"system": sys_merged, "question": req.user})
    answer = result.content

    # 建議改用 retriever.invoke（避免棄用警告）
    retrieved = retriever.invoke(req.user)
    sources = [d.page_content for d in retrieved]

    return {
        "answer": answer,
        "sources": sources,
        "k": 4,
        "mmr": {"fetch_k": 20, "lambda_mult": 0.2},
        "model": req.model,
        "embed_model": EMBED_MODEL,
    }

@app.post("/add_document")
def add_document(text: str):
    vector = embeddings.embed_query(text)
    payload = [("id-new", vector, {"text": text})]
    index.upsert(vectors=payload)
    return {"status": "ok"}

@app.post("/add_documents")
def add_documents(texts: List[str]):
    vectors = embeddings.embed_documents(texts)
    payload = [(f"id-{i}", v, {"text": t}) for i, (t, v) in enumerate(zip(texts, vectors))]
    index.upsert(vectors=payload)
    return {"status": "ok"}

@app.delete("/delete_all_documents")
def clear_indexes():
    index.delete(delete_all=True)
    return {"status": "all vectors deleted"}

@app.delete("/delete_by_ids")
def delete_by_ids(req: DeleteRequest):
    index.delete(ids=req.ids)
    return {"status": f"deleted {len(req.ids)} vectors"}

@app.get("/list_stats")
def list_stats():
    stats = index.describe_index_stats()
    # Pinecone SDK 版本不同，可能不是純 dict；保險處理：
    try:
        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        return JSONResponse(content=stats)
    except Exception:
        import json
        return JSONResponse(content=json.loads(json.dumps(stats, default=str)))
    
@app.get("/list_documents")
def list_documents(limit: int = 100):
    """
    列出 Pinecone 中前 100 筆文件。
    """
    try:
        res = index.query(
            vector=[0.0] * DIMENSION,  # 用一個 dummy 向量查詢
            top_k=limit,
            include_metadata=True
        )
        docs = [
            {"id": m["id"], "text": m["metadata"].get("text", "")[:80]}  # 只取前 80 字摘要
            for m in res.get("matches", [])
        ]
        return {"documents": docs}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)