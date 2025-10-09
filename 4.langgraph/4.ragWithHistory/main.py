# rag_langgraph_messages.py
from typing import List, Dict, Any, TypedDict
from pathlib import Path
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

# =========================
# (可選) 自動建立範例法律條文
# =========================
DOC_DIR = Path("./docs")
DOC_DIR.mkdir(parents=True, exist_ok=True)
LAW_FILE = DOC_DIR / "law_sample.txt"

# =========================
# 狀態（MessagesState + 額外欄位）
# =========================
class ChatRAGState(MessagesState, total=False):
    # MessagesState 內含: messages: List[BaseMessage]
    context: str   # 這輪檢索到的參考內容（方便印出）
    answer: str    # 這輪回答（方便印出）

# =========================
# 向量庫（Chroma, 持久化）
# =========================
PERSIST_DIR = "./chroma_db"

def load_docs_from_folder(folder: Path) -> List[Document]:
    docs: List[Document] = []
    for f in folder.glob("*.txt"):
        content = f.read_text(encoding="utf-8", errors="ignore")
        chunks = [x.strip() for x in content.split("\n\n") if x.strip()]
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk": i+1}))
    return docs

def get_or_build_chroma(docs: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )
    if not Path(PERSIST_DIR).exists() or not any(Path(PERSIST_DIR).iterdir()):
        vs = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
        vs.persist()
        return vs
    else:
        return Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)

DOCS = load_docs_from_folder(DOC_DIR)
VSTORE = get_or_build_chroma(DOCS)

# =========================
# 工具：抓「最後一則 Human 問句」
# =========================
def last_user_question(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content
    return ""

# =========================
# 節點：檢索（用最後一則 Human 問句）
# =========================
def retrieve(state: ChatRAGState) -> Dict[str, Any]:
    q = last_user_question(state.get("messages", []))
    results = VSTORE.similarity_search(q, k=3)
    joined = "\n\n".join(
        [f"[{i+1}] ({d.metadata.get('source')}#${d.metadata.get('chunk')})\n{d.page_content}"
         for i, d in enumerate(results)]
    )
    # 只更新本輪 context；不改 messages（避免把檢索片段灌進歷史）
    return {"context": joined}

# =========================
# 節點：回答（把 context 當 System，並 trim 歷史）
# =========================
def answer(state: ChatRAGState) -> Dict[str, Any]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
    ctx = state.get("context", "")
    history = state.get("messages", [])

    # 把本輪 context 包成 System 指令附在歷史尾端（僅送進模型，不寫回長期歷史）
    messages_with_ctx: List[BaseMessage] = history + [
        SystemMessage(content=(
            "你是一位嚴謹的法律助教。請僅根據『參考內容』回答問題；"
            "若無充分依據，請明確說「資料中未載明」。\n\n"
            f"[參考內容]\n{ctx}"
        ))
    ]

    trimmed = trim_messages(
        messages=messages_with_ctx,
        max_tokens=4000,
        token_counter=count_tokens_approximately,  # 關鍵：提供 token 計數器
        strategy="last",
        include_system=True,
        start_on="human",
    )

    resp = llm.invoke(trimmed)

    # 只把 AI 回覆寫回 messages（讓 checkpoint 記住對話）
    return {"messages": [AIMessage(content=resp.content)], "answer": resp.content}

# =========================
# 建圖 + checkpoint
# =========================
g = StateGraph(ChatRAGState)
g.add_node("Retrieve", retrieve)
g.add_node("Answer", answer)
g.add_edge(START, "Retrieve")
g.add_edge("Retrieve", "Answer")
g.add_edge("Answer", END)

memory = MemorySaver()  # 可換 SqliteSaver 讓歷史落盤
app = g.compile(checkpointer=memory)

# =========================
# REPL：每輪只丟本輪的人類訊息；歷史由 checkpoint 補上
# =========================
def repl():
    print("==== LangGraph RAG（Chroma）+ Checkpoint + Trim — 輸入 exit 離開 ====")
    print(f"已索引文件：{', '.join(sorted({d.metadata['source'] for d in DOCS}))}")

    thread_id = "law-chat-1"  # 用同一 thread_id 保留歷史
    while True:
        try:
            q = input("\n你> ").strip()
            if q.lower() == "exit":
                print("Bye!")
                break
            if not q:
                continue

            out = app.invoke(
                {"messages": [HumanMessage(content=q)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            print("\n[參考內容]")
            print(out.get("context", "(無)"))
            print("\n[回答]")
            print(out.get("answer", "(無)"))

        except KeyboardInterrupt:
            print("\n(按 Ctrl+C 離開或輸入 exit)")
        except Exception as e:
            print(f"\n[錯誤] {e}")

if __name__ == "__main__":
    repl()