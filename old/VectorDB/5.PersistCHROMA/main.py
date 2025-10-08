from pathlib import Path
from typing import List
from texts import texts

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 

# === 路徑與設定 ===
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_store"         
MODEL_NAME = "nomic-embed-text"

def build_or_load_vectorstore(embeddings: OllamaEmbeddings, seed_texts: List[str]) -> Chroma:
    """
    若已有持久化：載入向量庫
    否則：用 seed_texts 建立向量庫（Chroma 0.4.x 之後會自動持久化）
    回傳：vs
    """
    if PERSIST_DIR.exists():
        print("偵測到既有向量庫，從磁碟載入…")
        vs = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )
        return vs

    print("未偵測到向量庫，建立新索引並保存…")
    vs = Chroma.from_texts(
        texts=seed_texts,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    # 不需要 vs.persist()：Chroma 0.4.x 起自動持久化
    print(f"已建立向量庫並保存到 {PERSIST_DIR.name}")
    return vs

def interactive_loop(vs: Chroma) -> None:
    print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")
    while True:
        query = input("\n請輸入查詢：").strip()
        if query.lower() in ("exit", "quit"):
            print("結束程式。")
            break

        docs = vs.similarity_search(query, k=3)
        print("\nTop-3 相似結果：")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d.page_content}")

embeddings = OllamaEmbeddings(model=MODEL_NAME)
vs = build_or_load_vectorstore(embeddings, texts)
interactive_loop(vs)