import json
from texts import texts
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# === 設定 ===
MODEL_NAME = "nomic-embed-text"

# === 建立向量庫 ===
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vs = Chroma.from_texts(texts=texts, embedding=embeddings)
print("向量庫已建立。")

# === 重複詢問使用者輸入 ===
print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")

while True:
    query = input("\n請輸入查詢：").strip()
    if query.lower() in ("exit", "quit"):
        print("結束程式。")
        break

    docs = vs.similarity_search_with_score(query, k=5)
    print("\nTop-5 相似結果：")
    for i, (d, score) in enumerate(docs, 1):
        print(f"{i}. {d.page_content} (相似度: {score})")