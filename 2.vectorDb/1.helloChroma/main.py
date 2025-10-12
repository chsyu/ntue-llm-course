from texts import texts
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma  

# === 設定 ===
MODEL_NAME = "nomic-embed-text"

# === 建立向量庫（不持久化） ===
embeddings = OllamaEmbeddings(model=MODEL_NAME)
vs = Chroma.from_texts(texts=texts, embedding=embeddings) 

print("向量庫已建立。")