from dotenv import load_dotenv
from texts import texts
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 讀取 .env
load_dotenv()

# === 設定 ===
EMBED_MODEL = "text-embedding-3-small"  # 或 "text-embedding-3-large"

# === 建立向量庫 ===
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)  # 會自動讀取 OPENAI_API_KEY
vs = Chroma.from_texts(texts=texts, embedding=embeddings)
print("向量庫已建立。")

# === 重複詢問使用者輸入 ===
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