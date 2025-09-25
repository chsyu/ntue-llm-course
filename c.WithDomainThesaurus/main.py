from pathlib import Path
from typing import List, Dict, Iterable, Tuple
from texts import texts

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from domain_thesaurus import DOMAIN_THESAURUS

# === 路徑設定 ===
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "faiss_store"  # LangChain 的持久化資料夾（index.faiss + index.pkl）

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
    return vs

# --- Query Expansion ---

def expand_query(q: str, thesaurus: Dict[str, List[str]], max_expansions: int = 4) -> List[str]:
    """
    依小型詞庫做查詢擴展：比對命中的 key，收集其同義詞，組成多個變體查詢。
    策略：原查詢 +（原查詢 + 同義詞）… 最多回傳 1 + max_expansions 個。
    """
    q_lower = q.lower()
    expansions: List[str] = []
    for key, syns in thesaurus.items():
        if key in q_lower:
            # 將關鍵詞的同義詞加入 expansions
            for s in syns:
                expansions.append(f"{q} {s}")
    # 去重、裁切
    seen = set()
    uniq = []
    for e in expansions:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
        if len(uniq) >= max_expansions:
            break
    # 確保包含原查詢
    return [q] + uniq

def dedup_docs(docs: Iterable, key=lambda d: d.page_content) -> List:
    """按照出現順序去重（以 page_content 為 key）。"""
    seen = set()
    result = []
    for d in docs:
        k = key(d)
        if k not in seen:
            result.append(d)
            seen.add(k)
    return result

def search_with_expansion(vs: FAISS, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.2) -> Tuple[List, List[str]]:
    """
    綜合檢索：
      1) 對原查詢跑一次 MMR（fetch_k > k，增加候選多樣性）
      2) 對擴展查詢（若有）做一般相似度檢索
      3) 合併去重後，截取前 k 筆
    回傳：(docs, used_queries)
    """
    used_queries = []
    all_docs: List = []

    # 1) 原查詢 MMR
    mmr_docs = vs.max_marginal_relevance_search(query, k=min(k, 3), fetch_k=fetch_k, lambda_mult=lambda_mult)
    if mmr_docs:
        all_docs.extend(mmr_docs)
        used_queries.append(query + " [MMR]")

    # 2) 擴展查詢
    expanded = expand_query(query, DOMAIN_THESAURUS)
    # 第 0 個是原查詢，已做 MMR，從第 1 個開始跑
    for eq in expanded[1:]:
        sub_docs = vs.similarity_search(eq, k=3)
        if sub_docs:
            all_docs.extend(sub_docs)
            used_queries.append(eq)

    # 3) 合併去重 + 截取
    merged = dedup_docs(all_docs)
    return merged[:k], used_queries

# --- 互動查詢迴圈 ---

def interactive_loop(vs: FAISS) -> None:
    print("\n輸入你的問題（輸入 'exit' 或 'quit' 可結束）：")
    while True:
        query = input("\n請輸入查詢：").strip()
        if query.lower() in ("exit", "quit"):
            print("結束程式。")
            break

        docs, used_queries = search_with_expansion(vs, query=query, k=10, fetch_k=20, lambda_mult=0.2)

        # 顯示用到的查詢（方便教學觀察擴展有無幫助）
        print("\n本次使用的查詢：")
        for q in used_queries:
            print(" -", q)

        print("\nTop-5 綜合相似結果：")
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d.page_content}")

# --- 主程式 ---

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vs = build_or_load_vectorstore(embeddings, texts)
    interactive_loop(vs)