from pathlib import Path
from typing import List
from texts import texts

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === 路徑設定 ===
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "faiss_store"       # LangChain 的持久化資料夾（index.faiss + index.pkl）

MODEL_NAME = "nomic-embed-text"

def build_or_load_vectorstore(embeddings: OllamaEmbeddings, seed_texts: List[str]) -> FAISS:
    """
    若已有持久化：載入向量庫
    否則：用 seed_texts 建立向量庫 + 保存到磁碟
    回傳：vs
    """
    if PERSIST_DIR.exists():
        print("偵測到既有向量庫，從磁碟載入…")
        vs = FAISS.load_local(
            folder_path=str(PERSIST_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,  # 本機開發可用；上線請評估風險
        )
        return vs

    print("未偵測到向量庫，建立新索引並保存…")
    vs = FAISS.from_texts(texts=seed_texts, embedding=embeddings)
    vs.save_local(str(PERSIST_DIR))
    print(f"已建立向量庫，共 {vs.index.ntotal} 筆，並保存到 {PERSIST_DIR.name}")
    return vs, seed_texts

def interactive_loop(vs: FAISS) -> None:
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