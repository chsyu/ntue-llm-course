import streamlit as st
import requests

st.set_page_config(page_title="RAG 管理介面", layout="wide")
st.title("📚 RAG (Pinecone + OpenAI) 管理介面")

# 可在側邊欄調整 API 位址；預設用 127.0.0.1（避免 localhost 走 IPv6）
API_BASE = "http://127.0.0.1:5000"

def call_api(method: str, path: str, **kwargs):
    """統一呼叫 API，加入 timeout 與錯誤顯示。"""
    url = f"{API_BASE}{path}"
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        # 先顯示狀態碼
        st.caption(f"HTTP {resp.status_code} ← {url}")
        # 優先嘗試 JSON；不行就顯示文本
        try:
            return resp.json()
        except Exception:
            st.warning("⚠️ 回應非 JSON，以下顯示原始文字：")
            st.code(resp.text)
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 請求失敗：{e}")
        return None


# ========== Chat 區塊 ==========
st.header("💬 問答")
query = st.text_input("輸入問題：")
if st.button("送出查詢"):
    if not query.strip():
        st.warning("請先輸入問題。")
    else:
        data = call_api("POST", "/chat", json={"user": query})
        if data:
            st.success(data.get("answer", ""))
            st.write("### 🔎 來源")
            for i, src in enumerate(data.get("sources", []), 1):
                st.write(f"[{i}] {src}")

# ========== 新增文件 ==========
st.header("➕ 新增文件")
new_doc = st.text_area("輸入單一文件：")
if st.button("新增單筆文件"):
    data = call_api("POST", "/add_document", params={"text": new_doc})
    if data: st.success(data)

multi_docs = st.text_area("輸入多筆文件（每行一筆）：")
if st.button("新增多筆文件"):
    docs = [line.strip() for line in multi_docs.splitlines() if line.strip()]
    if not docs:
        st.warning("請先輸入至少一筆內容。")
    else:
        data = call_api("POST", "/add_documents", json=docs)
        if data: st.success(data)

# ========== 刪除文件 ==========
st.header("🗑 刪除文件")
if st.button("刪除全部文件"):
    data = call_api("DELETE", "/delete_all_documents")
    if data: st.success(data)

delete_ids = st.text_input("輸入要刪除的文件 ID（逗號分隔，如 id-1,id-5）")
if st.button("刪除指定文件"):
    ids = [x.strip() for x in delete_ids.split(",") if x.strip()]
    if not ids:
        st.warning("請先輸入至少一個 ID。")
    else:
        data = call_api("DELETE", "/delete_by_ids", json={"ids": ids})
        if data: st.success(data)

# ========== 索引狀態 ==========
st.header("📊 索引狀態")
if st.button("查看索引統計"):
    data = call_api("GET", "/list_stats")
    if data: st.json(data)
    
st.header("📑 文件清單")
limit = st.number_input("顯示幾筆文件", min_value=1, max_value=100, value=10)
if st.button("列出文件"):
    data = call_api("GET", f"/list_documents?limit={limit}")
    if data: st.json(data)