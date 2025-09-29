from dotenv import load_dotenv
import os
from texts import texts
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# --- 載入 API 金鑰 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
DIMENSION = 1536
INDEX_NAME = "demo-index"

# --- 初始化 Pinecone + Embedding ---
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

# --- 如果索引是空的就上傳 texts ---
if index.describe_index_stats().get("total_vector_count", 0) == 0:
    print("索引為空，上傳資料…")
    vectors = [
        (f"id-{i}", embeddings.embed_query(t), {"text": t})
        for i, t in enumerate(texts)
    ]
    index.upsert(vectors=vectors)
    print("已完成上傳")
else:
    print("索引已有資料，跳過上傳")

# --- 查詢迴圈 ---
print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")
while True:
    query = input("\n請輸入查詢：").strip()
    if query.lower() in ("exit", "quit"):
        print("結束程式。")
        break

    qvec = embeddings.embed_query(query)
    res = index.query(vector=qvec, top_k=3, include_metadata=True)
    matches = res.get("matches", [])
    print("\nTop-3 相似結果：")
    for i, m in enumerate(matches, 1):
        print(f"{i}. score={m['score']:.4f} | {m['metadata'].get('text')}")

# index.delete(delete_all=True)