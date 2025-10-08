from texts import texts
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma  # ← 換成 Chroma

# === 設定 ===
MODEL_NAME = "nomic-embed-text"

# 你的 texts: List[str]（每個元素是一篇長文）
# 1) 先切 chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,        # 起手式
    chunk_overlap=2,     # 15%
    add_start_index=True,  # 方便回溯原文位置
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 中文友善的遞迴分隔
)
docs = splitter.create_documents(texts)
print(f"原始文本共 {len(texts)} 篇")
print(f"切分成 {len(docs)} 個 chunks")
print("前 3 個 chunks：")
for d in docs[:3]:
    print(d)    

# === 建立向量庫（不持久化） ===
embeddings = OllamaEmbeddings(model=MODEL_NAME)
# vs = Chroma.from_texts(texts=texts, embedding=embeddings)  # ← 無 
vs = Chroma.from_documents(documents=docs, embedding=embeddings)
# persist_directory

print("向量庫已建立。")