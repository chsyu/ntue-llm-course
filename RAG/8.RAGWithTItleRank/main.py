# filename: app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import os, glob, re

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# ========== 可選 PDF 載入 ==========
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ========== 全域設定 ==========
LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"
EMBED_MODEL = "nomic-embed-text"
INDEX_DIR = "faiss_index"

app = FastAPI(title="RAG (FAISS) with Title Soft-Boost Fusion")
VECTOR_STORE: Optional[FAISS] = None
CURRENT_EMBED_MODEL = EMBED_MODEL

# ========== Schemas ==========
class ChatRequest(BaseModel):
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    user: str

class BuildIndexFromPathsReq(BaseModel):
    paths: List[str]
    # 這裡保留參數，但實作改為更穩健的「標題行」切分與抽取
    enable_regex_boundary: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 80
    embed_model: str = EMBED_MODEL
    save_path: str = INDEX_DIR

class RagChatReq(BaseModel):
    question: str
    k: int = 3
    model: str = LLM_MODEL
    system: Optional[str] = DEFAULT_SYSTEM_PROMPT
    pool_k: int = 24          # 候選池大小（建議 3~5 倍 k）

class ComplianceAskReq(BaseModel):
    question: str
    k: int = 5
    model: str = LLM_MODEL
    system: Optional[str] = "你是審慎的合規助理，嚴禁臆測，只能根據條文回答。"
    pool_k: int = 36

class IndexPathReq(BaseModel):
    path: str = INDEX_DIR
    embed_model: str = EMBED_MODEL

# ========== Helpers: 檔案、切分、索引儲存 ==========
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

# ---- 核心：更穩健的切分與抽取（支援全形/半形冒號）----

_HEADING_RE = re.compile(
    r"^\s*(ENERGY-\d{3})\s*[：:]\s*([^\n：:]{1,50})",  # group(1)=rule_id, group(2)=title
    flags=re.M
)

_BOUNDARY_RE = re.compile(
    r"(?=^\s*ENERGY-\d{3}\s*[：:])",  # 逐條前瞻切分
    flags=re.M
)

def extract_rule_and_title(text: str) -> (Optional[str], Optional[str]):
    m = _HEADING_RE.search(text)
    if not m:
        # 退化抽取 rule_id；title 留空
        rid = re.search(r"(ENERGY-\d{3})", text)
        return (rid.group(1) if rid else None, None)
    return m.group(1), m.group(2).strip()

def split_docs_by_heading(docs: List[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        parts = [m.start() for m in _BOUNDARY_RE.finditer(d.page_content)]
        if not parts:
            # 沒有明確標題行，整段保留（後續會用 fallback chunking）
            out.append(d)
            continue
        parts.append(len(d.page_content))
        for i in range(len(parts) - 1):
            seg = d.page_content[parts[i]:parts[i+1]].strip()
            if not seg:
                continue
            rid, title = extract_rule_and_title(seg)
            meta = dict(d.metadata)
            if rid:   meta["rule_id"] = rid
            if title: meta["title"]   = title
            out.append(Document(page_content=seg, metadata=meta))
    return out

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

# ========== Soft-Boost：標題 + 向量融合 ==========
def _sim_from_l2(dist: float) -> float:
    # FAISS L2 距離 → 相似度(0~1)
    return 1.0 / (1.0 + dist)

# --- 取代：用向量相似做「標題 soft-boost」 ---

from typing import Optional, List, Tuple
import math

# 用與索引相同的嵌入模型做即時嵌入
def _get_embedder():
    # 用你全域變數 CURRENT_EMBED_MODEL，與建庫一致
    return OllamaEmbeddings(model=CURRENT_EMBED_MODEL or EMBED_MODEL)

def _cosine_sim(a: List[float], b: List[float]) -> float:
    # 安全且快速的 cosine 相似度
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def _sim_from_l2(dist: float) -> float:
    # FAISS 相似度近似（距離→(0,1]）
    return 1.0 / (1.0 + dist)

def retrieve_with_title_vector_boost(
    vectorstore: FAISS,
    question: str,
    *,
    k: int,
    pool_k: int,
    w_body: float = 0.85,    # 內容相似的權重（主）
    w_title: float = 0.15,   # 標題相似的權重（輔）
) -> List[Document]:
    """
    1) 用 Body 向量索引取 pool_k 候選（含距離）
    2) 嵌入 query，一次完成
    3) 逐條候選：若 metadata 有 title，嵌入 title 並算 cosine(query, title)
    4) 融合分數：final = w_body * sim_body + w_title * sim_title
    5) 排序取前 k
    """
    hits: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(question, k=pool_k)

    # 只做一次 query 向量
    embedder = _get_embedder()
    q_vec = embedder.embed_query(question)

    # 避免重複嵌入同一個標題（常見於一條法條多個 chunk）
    title_vec_cache: dict[str, List[float]] = {}

    scored: List[Tuple[float, Document]] = []
    for doc, dist in hits:
        # 內容相似度（由距離轉相似度）
        sim_body = _sim_from_l2(dist)

        # 標題相似度（向量）
        title = (doc.metadata.get("title") or "").strip()
        if title:
            if title not in title_vec_cache:
                title_vec_cache[title] = embedder.embed_query(title)  # 可改 embed_documents([title])[0]
            t_vec = title_vec_cache[title]
            sim_title = _cosine_sim(q_vec, t_vec)  # [-1,1] → 一般會落在 0~1 區間

            # 保障落點（可選）：把負數截為 0
            if sim_title < 0:
                sim_title = 0.0
        else:
            sim_title = 0.0

        final = w_body * sim_body + w_title * sim_title
        # print(f"title: {title}, question: {question}, sim_title: {sim_title}, final: {final}")

        scored.append((final, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for final, doc in scored[:k]]


# ========== Lifecycle ==========
@app.on_event("startup")
def try_load_index_on_startup():
    if has_index_files(INDEX_DIR):
        try:
            _load_vector_store(INDEX_DIR, EMBED_MODEL)
            print(f"[startup] Loaded FAISS index from '{INDEX_DIR}' (embed_model={EMBED_MODEL})")
        except Exception as e:
            print(f"[startup] Found index but failed to load: {e}")

# ========== Endpoints ==========
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
    global VECTOR_STORE, CURRENT_EMBED_MODEL

    files = discover_files(req.paths)
    if not files:
        raise HTTPException(400, "找不到可讀取的檔案（支援 .txt/.md/.pdf）")

    raw_docs = load_docs(files)

    # 先嘗試「標題行切分 + 標題抽取」，若抓不到標題，才回退一般 chunk
    heading_docs = split_docs_by_heading(raw_docs)
    if heading_docs and any(d.metadata.get("rule_id") for d in heading_docs):
        docs_for_embed = heading_docs
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
        "docs_after_split": len(docs_for_embed),
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

    pool_k = max(req.pool_k, req.k * 4)  # 把候選抓多一點提高召回
    docs = retrieve_with_title_vector_boost(
        VECTOR_STORE,
        req.question,
        k=req.k,
        pool_k=max(req.pool_k, req.k * 4),
        w_body=0.05,   # 向量內容主導
        w_title=0.95   # 標題向量輔助（可調到 0.2~0.3 看場景）
    )
    context_text = join_docs(docs)
    # print(f"檢索到 {len(docs)} 篇文件，共 {len(context_text)} 字元")
    # print(docs_as_sources(docs))

    sys_merged = merge_system(req.system)
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_merged),
        ("user",
         "請僅根據以下內容回答問題；若不足請明確說「缺乏資料」。\n\n"
         "=== 內容 ===\n{context}\n\n=== 問題 ===\n{question}")
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

    pool_k = max(req.pool_k, req.k * 5)
    docs = retrieve_with_title_soft_boost(
        VECTOR_STORE,
        req.question,
        k=req.k,
        pool_k=pool_k,
        w_vec=0.8,    # 合規情境可加強標題權重
        w_title=0.2,
        title_bonus=1.0
    )
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