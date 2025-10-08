from dotenv import load_dotenv
import os
from typing import List

from texts import texts
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

# 綁定到 public schema
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE).schema("public")

def build_or_load_vectorstore(seed_texts: List[str]) -> SupabaseVectorStore:
    vs = SupabaseVectorStore(
        client=supabase,
        table_name="documents",
        query_name="match_documents_v2",   # ← 改用新函式
        embedding=embeddings,
    )

    # 如果表是空的就塞資料
    count = supabase.table("documents").select("id", count="exact").limit(1).execute().count or 0
    if count == 0 and seed_texts:
        print("未偵測到資料，開始上傳 seed_texts …")
        vs.add_texts(texts=seed_texts, metadatas=[{} for _ in seed_texts])
        print("已完成上傳並建立向量。")
    else:
        print(f"偵測到既有資料（{count} 筆），直接使用。")
    return vs

def interactive_loop(vs: SupabaseVectorStore) -> None:
    print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")
    while True:
        query = input("\n請輸入查詢：").strip()
        if query.lower() in ("exit", "quit"):
            print("結束程式。")
            break
        docs = vs.similarity_search(query, k=3)
        print("\nTop-3 相似結果：")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d.page_content} | meta={d.metadata}")


vs = build_or_load_vectorstore(texts)
interactive_loop(vs)