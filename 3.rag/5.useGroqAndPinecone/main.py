from dotenv import load_dotenv
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from texts import texts  # 你的教材語料

# LangChain + Groq + Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBED_MODEL = "text-embedding-3-small"      # OpenAI embedding
LLM_MODEL   = "llama-3.1-8b-instant"               # Groq chat模型
INDEX_NAME  = "rag-demo-index"
DIMENSION   = 1536                          # text-embedding-3-small 對應 1536維
DEFAULT_SYSTEM_PROMPT = (
    ""
)

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

# 用 MMR 的 Retriever（更穩定）
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.2},
)

def format_docs(docs) -> str:
    return "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

# =====================
# Chain：檢索 → Prompt → LLM
# =====================
retriever_chain = RunnablePassthrough.assign(
    context=lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
)

prompt = ChatPromptTemplate.from_messages([
    ("system"
    #  "你是一個精煉且忠實的助教。\n"
    #  "你的回答**必須完全依據提供的內容**，不得自行推測或引入外部知識。\n"
    #  "如果內容不足以回答，請直接回答：『我不知道』。"
     ),
    ("user",
     ""
     "【內容】\n{context}\n\n"
     "【問題】\n{question}")
])

llm = ChatGroq(model=LLM_MODEL, temperature=0.2, groq_api_key=GROQ_API_KEY)

rag_chain = retriever_chain | prompt | llm

# =====================
# FastAPI
# =====================
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

app = FastAPI(title="RAG (Pinecone + OpenAI)")

@app.post("/chat")
def chat(req: ChatRequest):
    sys_merged = (
        DEFAULT_SYSTEM_PROMPT
        if req.system == DEFAULT_SYSTEM_PROMPT
        else f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{req.system or ''}"
    )

    # 執行完整 RAG 鏈
    result = rag_chain.invoke({"system": sys_merged, "question": req.user})
    answer = result.content

    retrieved = retriever.get_relevant_documents(req.user)
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
    vector = embeddings.embed_query(text)  # embed_query 產生單一向量
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
    """
    刪除整個 Pinecone index 的所有向量。
    """
    index.delete(delete_all=True)
    return {"status": "all vectors deleted"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)