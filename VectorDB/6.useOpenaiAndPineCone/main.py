# main.py
from dotenv import load_dotenv
import os, time
from typing import Dict, Any, List
from texts import texts
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert OPENAI_API_KEY and PINECONE_API_KEY, "請在 .env 設定 OPENAI_API_KEY / PINECONE_API_KEY"

# ==== 設定 ====
EMBED_MODEL = "text-embedding-3-small"  # 1536 維；若改 -large → 改 DIMENSION=3072
DIMENSION = 1536
INDEX_NAME = "demo-index"
NAMESPACE = "demo-v1"                   
TOP_K = 3

# ==== 初始化 ====
pc = Pinecone(api_key=PINECONE_API_KEY)
emb = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

def ensure_index(name: str, dim: int):
    existing = {i["name"] for i in pc.list_indexes().get("indexes", [])}
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def upsert_if_needed(index_name: str, ns: str, docs: List[str]):
    index = pc.Index(index_name)

    stats: Dict[str, Any] = index.describe_index_stats()
    count = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)

    if count > 0:
        print(f"namespace={ns} 已有 {count} 筆，跳過嵌入與上傳。")
        return

    print(f"namespace={ns} 目前沒有資料，開始上傳 …")
    print(f"開始嵌入 {len(docs)} 筆文本 …")
    vectors = emb.embed_documents(docs)  # 這裡才會呼叫 OpenAI API
    payload = [(f"doc-{i+1}", v, {"text": t}) for i, (t, v) in enumerate(zip(docs, vectors))]
    index.upsert(vectors=payload, namespace=ns)
    print("上傳完成。")

def query_loop(index_name: str, ns: str):
    index = pc.Index(index_name)
    print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")
    while True:
        q = input("\n請輸入查詢：").strip()
        if q.lower() in ("exit", "quit"):
            print("結束程式。"); break
        qvec = emb.embed_query(q)  # 查詢向量（會呼叫 OpenAI，一次一筆）
        res = index.query(vector=qvec, top_k=TOP_K, include_metadata=True, namespace=ns)
        matches = res.get("matches", [])
        if not matches:
            print("沒有找到相似結果。"); continue
        print(f"\nTop-{TOP_K} 相似結果：")
        for i, m in enumerate(matches, 1):
            print(f"{i}. id={m['id']} score={m['score']:.4f} | {m['metadata'].get('text')}")


ensure_index(INDEX_NAME, DIMENSION)
upsert_if_needed(INDEX_NAME, NAMESPACE, texts)
print("向量索引已就緒。")
query_loop(INDEX_NAME, NAMESPACE)