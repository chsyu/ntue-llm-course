from texts import texts
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === 設定 ===
MODEL_NAME = "nomic-embed-text"

# === 建立或載入向量庫 ===
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vs = FAISS.from_texts(texts=texts, embedding=embeddings)

print("向量庫已建立。")
