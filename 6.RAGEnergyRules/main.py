# filename: main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, glob, re

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# 可選載入器（有裝 pypdf 才能用）
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ===== 基本設定 =====
LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"
EMBED_MODEL = "nomic-embed-text"

# 索引檔儲存資料夾（會產生 index.faiss + index.pkl）
INDEX_DIR = "faiss_index"

app = FastAPI(title="LC + Ollama: Regex-split RAG with FAISS persistence")
VECTOR_STORE: Optional[FAISS] = None
CURRENT_EMBED_MODEL = EMBED_MODEL  # 記錄目前索引使用的 embedding 模型

# ===== Pydantic Schemas =====
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

class BuildIndexFromPathsReq(BaseModel):
    paths: List[str]                    # 檔案或資料夾路徑，支援萬用字元 *.txt
    # 若 regex_boundary 啟用，將先以 regex 對齊「條文邊界」切分；
    # 否則採用一般 chunk（chunk_size / chunk_overlap）
    enable_regex_boundary: bool = True
    # 斷點偵測：用「前瞻分割」把每個匹配項視為一段的開始
    # 預設針對「ENERGY-001：」這種條碼格式；你可換成法律條碼
    regex_boundary: str = r"(?=(ENERGY-\d{3}：))"
    # 從每段內容抽出 rule_id 的 regex（需含一個 group）
    regex_rule_id: str = r"(ENERGY-\d{3})"
    # 落回一般字數切分的備援（PDF 或 regex 不適用時）
    chunk_size: int = 500
    chunk_overlap: int = 80
    embed_model: str = EMBED_MODEL
    save_path: str = INDEX_DIR         # 建索引後存檔的資料夾

class RagChatReq(BaseModel):
    question: str
    k: int = 3
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    search_type: str = "similarity"  # 或 "mmr"

class ComplianceAskReq(BaseModel):
    question: str
    k: int = 5
    model: str = LLM_MODEL
    system: Optional[str] = "你是審慎的合規助理，嚴禁臆測，只能根據條文回答。"
    search_type: str = "similarity"

class IndexPathReq(BaseModel):
    path: str = INDEX_DIR
    embed_model: str = EMBED_MODEL  # 載入時必須提供對應的 embedding 模型

# ===== Helpers =====
def merge_system(user_system: Optional[str]) -> str:
    if not user_system or user_system.strip() == DEFAULT_SYSTEM_PROMPT:
        return DEFAULT_SYSTEM_PROMPT
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{user_system.strip()}"

def discover_files(paths: List[str]) -> List[str]:
    """展開資料夾與萬用字元，僅保留支援的副檔名。"""
    found = []
    exts = {".txt", ".md", ".pdf"} if HAS_PDF else {".txt", ".md"}
    for p in paths:
        if os.path.isdir(p):
            for ext in exts:
                found.extend(glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True))
        else:
            found.extend(glob.glob(p))
    found = [f for f in found if os.path.splitext(f)[1].lower() in exts]
    return sorted(set(found))

def load_docs(filepaths: List[str]) -> List[Document]:
    """依副檔名選擇對應載入器，回傳 Document 陣列。"""
    docs: List[Document] = []
    for fp in filepaths:
        ext = os.path.splitext(fp)[1].lower()
        if ext in (".txt", ".md"):
            docs.extend(TextLoader(fp, encoding="utf-8").load())
        elif ext == ".pdf":
            if not HAS_PDF:
                raise RuntimeError("未安裝 pypdf，無法讀取 PDF，請 `pip install pypdf`")
            docs.extend(PyPDFLoader(fp).load())
    return docs

def regex_split_document(
    text: str,
    source: str,
    regex_boundary: str,
    regex_rule_id: str
) -> List[Document]:
    """
    以 regex 邊界切分：
    - regex_boundary 需為「前瞻分割」形式，例如 (?=(ENERGY-\d{3}：))
      表示每次匹配都視為下一段的「開始」。
    - regex_rule_id 用來從段內抽取條碼，例如 (ENERGY-\d{3})
    """
    # 先用 boundary 切分（保留分隔詞本身為每段的起頭）
    parts = re.split(regex_boundary, text)
    # re.split 會產生： [前置內容, 命中1, 命中1_後續, 命中2, 命中2_後續, ...]
    # 我們要把 (命中X + 命中X_後續) 合併成一段
    chunks: List[str] = []
    if len(parts) <= 1:
        # 沒有命中，整段視為一個 chunk
        chunks = [text]
    else:
        # parts[0] 可能是 boundary 前的前置內容（通常空或說明）
        # 之後每兩個一組：marker + body
        for i in range(1, len(parts), 2):
            head = parts[i]
            body = parts[i+1] if (i + 1) < len(parts) else ""
            chunks.append((head + body).strip())

    docs: List[Document] = []
    for ch in chunks:
        rule_id = None
        m = re.search(regex_rule_id, ch)
        if m:
            rule_id = m.group(1)
        meta = {"source": source}
        if rule_id:
            meta["rule_id"] = rule_id
        docs.append(Document(page_content=ch.strip(), metadata=meta))
    return docs

def split_docs_by_regex(
    docs: List[Document],
    regex_boundary: str,
    regex_rule_id: str
) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        out.extend(regex_split_document(
            text=d.page_content,
            source=d.metadata.get("source", ""),
            regex_boundary=regex_boundary,
            regex_rule_id=regex_rule_id,
        ))
    # 過濾空白
    out = [x for x in out if x.page_content.strip()]
    return out

def fallback_chunk(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def join_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def docs_as_sources(docs: List[Document], max_chars: int = 220) -> List[Dict[str, str]]:
    """把檢索到的文件，整理成 citations：優先顯示 rule_id，否則顯示檔名。"""
    out = []
    for d in docs:
        rule_id = d.metadata.get("rule_id")
        src = d.metadata.get("source", "")
        head = rule_id or os.path.basename(src) or "(unknown)"
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "…"
        out.append({"source": head, "snippet": snippet})
    return out

def has_index_files(path: str) -> bool:
    return os.path.exists(os.path.join(path, "index.faiss")) and \
           os.path.exists(os.path.join(path, "index.pkl"))

def _load_vector_store(path: str, embed_model: str):
    global VECTOR_STORE, CURRENT_EMBED_MODEL
    embeddings = OllamaEmbeddings(model=embed_model)
    # 允許反序列化（教學用途；正式環境請控管檔案來源）
    VECTOR_STORE = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    CURRENT_EMBED_MODEL = embed_model

# ===== Lifecycle =====
@app.on_event("startup")
def try_load_index_on_startup():
    """啟動時若有索引檔，自動載入（用預設 EMBED_MODEL）。"""
    if has_index_files(INDEX_DIR):
        try:
            _load_vector_store(INDEX_DIR, EMBED_MODEL)
            print(f"[startup] Loaded FAISS index from '{INDEX_DIR}' (embed_model={EMBED_MODEL})")
        except Exception as e:
            print(f"[startup] Found index but failed to load: {e}")

# ===== Endpoints =====
@app.get("/rag/status")
def rag_status():
    """檢查索引是否存在與載入狀態。"""
    present = has_index_files(INDEX_DIR)
    loaded = VECTOR_STORE is not None
    return {
        "files_present": present,
        "loaded_in_memory": loaded,
        "index_path": INDEX_DIR,
        "embed_model_in_memory": CURRENT_EMBED_MODEL if loaded else None
    }

@app.post("/chat")
def chat(req: ChatRequest):
    sys_merged = merge_system(req.system)
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user", "{question}")
    ])
    llm = ChatOllama(model=req.model, temperature=0.3)
    chain = prompt | llm
    result = chain.invoke({"question": req.user})
    return {"answer": result.content}

@app.post("/rag/build_index_from_paths")
def build_index_from_paths(req: BuildIndexFromPathsReq):
    """
    讀檔→（優先）Regex對齊條文邊界切分→（備援）一般chunk→向量化→建立FAISS→存檔
    """
    global VECTOR_STORE, CURRENT_EMBED_MODEL

    files = discover_files(req.paths)
    if not files:
        raise HTTPException(400, "找不到可讀取的檔案（支援 .txt/.md/.pdf）")

    raw_docs = load_docs(files)

    # Regex 對齊條文邊界（對 .txt/.md 效果最佳；PDF 視內容而定）
    if req.enable_regex_boundary:
        try:
            regex_docs = split_docs_by_regex(raw_docs, req.regex_boundary, req.regex_rule_id)
            # 若 regex 切不出東西，回退一般 chunk
            docs_for_embed = regex_docs if regex_docs else fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)
        except re.error as e:
            # regex 寫錯時回退
            print(f"[regex] invalid pattern: {e}; fallback to normal chunking.")
            docs_for_embed = fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)
    else:
        docs_for_embed = fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)

    embeddings = OllamaEmbeddings(model=req.embed_model)
    VECTOR_STORE = FAISS.from_documents(docs_for_embed, embeddings)
    CURRENT_EMBED_MODEL = req.embed_model

    # 儲存到磁碟（index.faiss + index.pkl）
    os.makedirs(req.save_path, exist_ok=True)
    VECTOR_STORE.save_local(req.save_path)

    return {
        "status": "ok",
        "files": files,
        "docs_after_regex_or_chunk": len(docs_for_embed),
        "embed_model": req.embed_model,
        "saved_to": req.save_path
    }

class IndexPathReq(BaseModel):
    path: str = INDEX_DIR
    embed_model: str = EMBED_MODEL

@app.post("/rag/save")
def rag_save(req: IndexPathReq):
    """將目前記憶體中的索引存檔。"""
    if VECTOR_STORE is None:
        raise HTTPException(400, "尚未建立/載入索引，無法存檔")
    os.makedirs(req.path, exist_ok=True)
    VECTOR_STORE.save_local(req.path)
    return {"status": "saved", "path": req.path, "embed_model": CURRENT_EMBED_MODEL}

@app.post("/rag/load")
def rag_load(req: IndexPathReq):
    """從磁碟載入索引到記憶體（embed_model 必須與建索引時一致）。"""
    if not has_index_files(req.path):
        raise HTTPException(400, f"索引檔不存在：{req.path}")
    _load_vector_store(req.path, req.embed_model)
    return {"status": "loaded", "path": req.path, "embed_model": req.embed_model}

@app.post("/rag/chat")
def rag_chat(req: RagChatReq):
    if VECTOR_STORE is None:
        raise HTTPException(400, "索引未載入。請先 /rag/build_index_from_paths 或 /rag/load")

    retriever = VECTOR_STORE.as_retriever(
        search_type=req.search_type,
        search_kwargs={"k": req.k}
    )
    sys_merged = merge_system(req.system)

    # 嚴格 RAG：只根據檢索內容回答，不足就說缺資料
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user",
         "請根據以下提供的內容回答問題。\n\n"
         "=== 檢索到的內容 ===\n{context}\n\n"
         "=== 問題 ===\n{question}\n\n"
         "如果內容不足以回答問題，請明確指出「缺乏資料」並不要自行臆測。")
    ])

    llm = ChatOllama(model=req.model, temperature=0.2)
    parser = StrOutputParser()

    # 先抓 docs 以便回傳 citations
    docs = retriever.get_relevant_documents(req.question)
    context_text = join_docs(docs)

    chain = {"question": RunnablePassthrough(), "context": RunnablePassthrough()} | prompt | llm | parser
    answer = chain.invoke({"question": req.question, "context": context_text})
    citations = docs_as_sources(docs)

    return {"answer": answer, "sources": citations}

@app.post("/rag/ask_compliance")
def ask_compliance(req: ComplianceAskReq):
    if VECTOR_STORE is None:
        raise HTTPException(400, "索引未載入。請先 /rag/build_index_from_paths 或 /rag/load")

    retriever = VECTOR_STORE.as_retriever(
        search_type=req.search_type,
        search_kwargs={"k": req.k}
    )
    sys_merged = req.system  # 或與 DEFAULT_SYSTEM_PROMPT 合併

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user",
         "你將扮演『能源/法規合規助理』。請只根據提供的條文段落回答以下問題。\n"
         "若條文不足以判定，請明確說明『缺乏資料』並列出仍需的資訊。\n\n"
         "=== 檢索到的條文段落 ===\n{context}\n\n"
         "=== 問題/案場說明 ===\n{question}\n\n"
         "請依序輸出：\n"
         "1) 判定：符合 / 可能不符合 / 無法判定（缺乏資料）\n"
         "2) 依據條碼與摘錄（若有 rule_id 請標示，例如 ENERGY-012）\n"
         "3) 理由（比對門檻：樓地板面積、契約容量、設備等級、設定溫度、是否需文件…）\n"
         "4) 若不確定，列出需要補充的具體資料清單")
    ])

    llm = ChatOllama(model=req.model, temperature=0.2)
    parser = StrOutputParser()

    docs = retriever.get_relevant_documents(req.question)
    context_text = join_docs(docs)

    chain = {"question": RunnablePassthrough(), "context": RunnablePassthrough()} | prompt | llm | parser
    answer = chain.invoke({"question": req.question, "context": context_text})
    citations = docs_as_sources(docs)

    return {"answer": answer, "sources": citations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port= 5000, reload=True)