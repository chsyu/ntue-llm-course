# filename: app.py
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

# ====== 可選載入器（PDF）======
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ====== 參數 ======
LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "faiss_index"

app = FastAPI(title="Regex-split RAG with FAISS + Metadata Rerank (vector-only)")
VECTOR_STORE: Optional[FAISS] = None
CURRENT_EMBED_MODEL = EMBED_MODEL  # 記錄向量庫使用的 embedding 模型

# ====== Schemas ======
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

class BuildIndexFromPathsReq(BaseModel):
    paths: List[str]
    enable_regex_boundary: bool = True
    regex_boundary: str = r"(?=(ENERGY-\d{3}：))"   # 每條的起始（前瞻）
    regex_rule_id: str = r"(ENERGY-\d{3})"          # 抽出條碼
    chunk_size: int = 500
    chunk_overlap: int = 80
    embed_model: str = EMBED_MODEL
    save_path: str = INDEX_DIR

class RagChatReq(BaseModel):
    question: str
    k: int = 3
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    search_type: str = "mmr"  # 建議用 mmr 提升多樣性

class ComplianceAskReq(BaseModel):
    question: str
    k: int = 5
    model: str = LLM_MODEL
    system: Optional[str] = "你是審慎的合規助理，嚴禁臆測，只能根據條文回答。"
    search_type: str = "mmr"

class IndexPathReq(BaseModel):
    path: str = INDEX_DIR
    embed_model: str = EMBED_MODEL

# ====== Helpers ======
def merge_system(user_system: Optional[str]) -> str:
    if not user_system or user_system.strip() == DEFAULT_SYSTEM_PROMPT:
        return DEFAULT_SYSTEM_PROMPT
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n[用戶補充]\n{user_system.strip()}"

def discover_files(paths: List[str]) -> List[str]:
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

def regex_split_document(text: str, source: str, regex_boundary: str, regex_rule_id: str) -> List[Document]:
    parts = re.split(regex_boundary, text)
    chunks: List[str] = []
    if len(parts) <= 1:
        chunks = [text]
    else:
        for i in range(1, len(parts), 2):
            head = parts[i]
            body = parts[i+1] if (i + 1) < len(parts) else ""
            chunks.append((head + body).strip())

    docs: List[Document] = []
    for ch in chunks:
        meta = {"source": source}
        # 抽 rule_id
        m = re.search(regex_rule_id, ch)
        if m:
            meta["rule_id"] = m.group(1)
        # 抽 title（ENERGY-123：XXX）
        tm = re.search(r"ENERGY-\d{3}：([^：\n]{1,30})", ch)
        if tm:
            meta["title"] = tm.group(1).strip()
        docs.append(Document(page_content=ch.strip(), metadata=meta))
    return docs

def split_docs_by_regex(docs: List[Document], regex_boundary: str, regex_rule_id: str) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        out.extend(regex_split_document(d.page_content, d.metadata.get("source", ""), regex_boundary, regex_rule_id))
    return [x for x in out if x.page_content.strip()]

def fallback_chunk(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def join_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

def docs_as_sources(docs: List[Document], max_chars: int = 220) -> List[Dict[str, str]]:
    out = []
    for d in docs:
        rid = d.metadata.get("rule_id")
        title = d.metadata.get("title")
        src = d.metadata.get("source", "")
        head = " / ".join([x for x in [rid, title] if x]) or os.path.basename(src) or "(unknown)"
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "…"
        out.append({"source": head, "snippet": snippet})
    return out

def has_index_files(path: str) -> bool:
    return os.path.exists(os.path.join(path, "index.faiss")) and os.path.exists(os.path.join(path, "index.pkl"))

def _load_vector_store(path: str, embed_model: str):
    global VECTOR_STORE, CURRENT_EMBED_MODEL
    embeddings = OllamaEmbeddings(model=embed_model)
    VECTOR_STORE = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    CURRENT_EMBED_MODEL = embed_model

# ====== Metadata 加權重排（vector-only）======
def expand_terms_for_weight(q: str) -> List[str]:
    """可依領域客製；此處示範熱回收同義詞。"""
    q = q.lower()
    terms = []
    if any(t in q for t in ["熱回收", "餘熱", "廢熱", "heat recovery"]):
        terms += ["熱回收", "餘熱回收", "廢熱回收", "余熱回收", "heat recovery"]
    if any(t in q for t in ["空壓", "壓縮空氣"]):
        terms += ["空壓", "壓縮空氣"]
    if "鍋爐" in q:
        terms += ["鍋爐"]
    # 也保留原問句關鍵字（最簡單：直接丟原句，或自行做簡單抽詞）
    return list(dict.fromkeys(terms)) or [q]

def metadata_rerank(docs: List[Document], question: str, top_n: int) -> List[Document]:
    """根據 rule_id / title / 內文命中給加權，重新排序。"""
    q = question.lower()
    bonus_terms = expand_terms_for_weight(q)

    def score(d: Document) -> float:
        s = 0.0
        rid = (d.metadata.get("rule_id") or "").lower()
        title = (d.metadata.get("title") or "").lower()
        text = d.page_content.lower()

        # 若問題直接包含條碼（如 ENERGY-073），給高分
        if rid and rid in q:
            s += 5.0
        # 標題命中強加權（例如「熱回收」）
        if any(t in title for t in bonus_terms):
            s += 3.0
        # 內文命中弱加權
        if any(t in text for t in bonus_terms):
            s += 1.0

        # 可以視情況對「空壓/鍋爐」再微調
        if any(t in q for t in ["空壓", "壓縮空氣"]) and ("空壓" in title or "壓縮空氣" in text):
            s += 1.5
        if ("鍋爐" in q) and ("鍋爐" in title or "鍋爐" in text):
            s += 1.0

        return s

    # 以加權分數排序，取前 top_n
    docs_sorted = sorted(docs, key=score, reverse=True)
    return docs_sorted[:top_n]

# ====== Lifecycle ======
@app.on_event("startup")
def try_load_index_on_startup():
    if has_index_files(INDEX_DIR):
        try:
            _load_vector_store(INDEX_DIR, EMBED_MODEL)
            print(f"[startup] Loaded FAISS index from '{INDEX_DIR}' (embed_model={EMBED_MODEL})")
        except Exception as e:
            print(f"[startup] Found index but failed to load: {e}")

# ====== Endpoints ======
@app.get("/rag/status")
def rag_status():
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
    """讀檔 → Regex 切分（或回退一般 chunk）→ 向量化 → 建 FAISS → 存檔"""
    global VECTOR_STORE, CURRENT_EMBED_MODEL

    files = discover_files(req.paths)
    if not files:
        raise HTTPException(400, "找不到可讀取的檔案（支援 .txt/.md/.pdf）")

    raw_docs = load_docs(files)

    if req.enable_regex_boundary:
        try:
            regex_docs = split_docs_by_regex(raw_docs, req.regex_boundary, req.regex_rule_id)
            docs_for_embed = regex_docs if regex_docs else fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)
        except re.error as e:
            print(f"[regex] invalid pattern: {e}; fallback to normal chunking.")
            docs_for_embed = fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)
    else:
        docs_for_embed = fallback_chunk(raw_docs, req.chunk_size, req.chunk_overlap)

    embeddings = OllamaEmbeddings(model=req.embed_model)
    VECTOR_STORE = FAISS.from_documents(docs_for_embed, embeddings)
    CURRENT_EMBED_MODEL = req.embed_model

    os.makedirs(req.save_path, exist_ok=True)
    VECTOR_STORE.save_local(req.save_path)

    return {
        "status": "ok",
        "files": files,
        "docs_after_regex_or_chunk": len(docs_for_embed),
        "embed_model": req.embed_model,
        "saved_to": req.save_path
    }

@app.post("/rag/save")
def rag_save(req: IndexPathReq):
    if VECTOR_STORE is None:
        raise HTTPException(400, "尚未建立/載入索引，無法存檔")
    os.makedirs(req.path, exist_ok=True)
    VECTOR_STORE.save_local(req.path)
    return {"status": "saved", "path": req.path, "embed_model": CURRENT_EMBED_MODEL}

@app.post("/rag/load")
def rag_load(req: IndexPathReq):
    if not has_index_files(req.path):
        raise HTTPException(400, f"索引檔不存在：{req.path}")
    _load_vector_store(req.path, req.embed_model)
    return {"status": "loaded", "path": req.path, "embed_model": req.embed_model}

@app.post("/rag/chat")
def rag_chat(req: RagChatReq):
    if VECTOR_STORE is None:
        raise HTTPException(400, "索引未載入。請先 /rag/build_index_from_paths 或 /rag/load")

    # 先擴大候選集合（mmr + k 值稍大）
    pool_k = max(req.k * 3, 12)
    vec_retriever = VECTOR_STORE.as_retriever(
        search_type=req.search_type,
        search_kwargs={"k": pool_k}
    )
    pool_docs = vec_retriever.get_relevant_documents(req.question)

    # 再依 metadata（rule_id/title/內文關鍵詞）加權重排，最後取前 k
    docs = metadata_rerank(pool_docs, req.question, top_n=req.k)
    context_text = join_docs(docs)

    sys_merged = merge_system(req.system)
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

    chain = {"question": RunnablePassthrough(), "context": RunnablePassthrough()} | prompt | llm | parser
    answer = chain.invoke({"question": req.question, "context": context_text})
    citations = docs_as_sources(docs)
    return {"answer": answer, "sources": citations}

@app.post("/rag/ask_compliance")
def ask_compliance(req: ComplianceAskReq):
    if VECTOR_STORE is None:
        raise HTTPException(400, "索引未載入。請先 /rag/build_index_from_paths 或 /rag/load")

    pool_k = max(req.k * 3, 12)
    vec_retriever = VECTOR_STORE.as_retriever(
        search_type=req.search_type,
        search_kwargs={"k": pool_k}
    )
    pool_docs = vec_retriever.get_relevant_documents(req.question)
    docs = metadata_rerank(pool_docs, req.question, top_n=req.k)
    context_text = join_docs(docs)

    sys_merged = req.system
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

    chain = {"question": RunnablePassthrough(), "context": RunnablePassthrough()} | prompt | llm | parser
    answer = chain.invoke({"question": req.question, "context": context_text})
    citations = docs_as_sources(docs)
    return {"answer": answer, "sources": citations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port= 5000, reload=True)