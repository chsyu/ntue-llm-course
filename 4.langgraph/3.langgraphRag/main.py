from typing import TypedDict, List, Dict, Any, Annotated
from pathlib import Path
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import Chroma

# 為了支持狀態累積的輔助函數
def add_to_list(x: list, y: list) -> list:
    """合併兩個列表的函數"""
    return x + y

load_dotenv()

# =========================
# 自動建立範例法律條文
# =========================
DOC_DIR = Path("./docs")
DOC_DIR.mkdir(parents=True, exist_ok=True)
LAW_FILE = DOC_DIR / "law_sample.txt"

# =========================
# 狀態定義 - 更符合 LangGraph 最佳實踐
# =========================
class RAGState(TypedDict):
    # 使用 Annotated 類型來定義狀態累積策略
    messages: Annotated[List, add_to_list]  # 對話歷史
    question: str  # 當前問題
    context: str  # 檢索到的上下文
    retrieved_docs: List[Document]  # 檢索到的文檔
    answer: str  # 最終答案

# =========================
# 構建／載入 Chroma 向量庫（持久化）
# =========================
PERSIST_DIR = "./chroma_db"

def load_docs_from_folder(folder: Path) -> List[Document]:
    docs: List[Document] = []
    for f in folder.glob("*.txt"):
        content = f.read_text(encoding="utf-8", errors="ignore")
        # 以空行切塊，可自行換成 LangChain TextSplitter
        chunks = [x.strip() for x in content.split("\n\n") if x.strip()]
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": f.name, "chunk": i+1}))
    return docs

def get_or_build_chroma(docs: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    # 若資料夾不存在或為空：建立；否則載入既有索引
    if not Path(PERSIST_DIR).exists() or not any(Path(PERSIST_DIR).iterdir()):
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        vs.persist()
        return vs
    else:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

DOCS = load_docs_from_folder(DOC_DIR)
VSTORE = get_or_build_chroma(DOCS)

# =========================
# 節點：檢索 - 正確的狀態傳遞方式
# =========================
def retrieve(state: RAGState) -> RAGState:
    """檢索相關文檔並更新狀態"""
    q = state.get("question", "")
    
    # 執行向量檢索
    results = VSTORE.similarity_search(q, k=3)
    
    # 格式化上下文
    joined = "\n\n".join(
        [f"[{i+1}] ({d.metadata.get('source')}#${d.metadata.get('chunk')})\n{d.page_content}"
         for i, d in enumerate(results)]
    )
    
    # 返回完整的狀態更新 - 這是關鍵！
    return {
        **state,  # 保留原有狀態
        "context": joined,  # 更新上下文
        "retrieved_docs": results,  # 保存檢索到的文檔
        "messages": [HumanMessage(content=f"檢索到 {len(results)} 個相關文檔")]  # 添加消息
    }

# =========================
# 節點：回答 - 正確的狀態傳遞方式
# =========================
def answer_with_groq(state: RAGState) -> RAGState:
    """基於上下文生成答案並更新狀態"""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
    
    q = state.get("question", "")
    ctx = state.get("context", "")
    
    # 構建完整的對話上下文
    system_msg = SystemMessage(content="""你是一位嚴謹的法律助教。請僅根據「參考內容」回答問題；
無充分依據時請明確說「資料中未載明」。重點條列與條次引用可加分。
請用繁體中文作答，並在末行標示引用片段編號（如：[1][3]）。""")
    
    user_msg = HumanMessage(content=f"""[參考內容]
{ctx}

[問題]
{q}""")
    
    # 調用 LLM
    response = llm.invoke([system_msg, user_msg])
    
    # 返回完整的狀態更新
    return {
        **state,  # 保留原有狀態
        "answer": response.content,  # 更新答案
        "messages": [
            AIMessage(content=f"基於 {len(state.get('retrieved_docs', []))} 個文檔生成回答"),
            AIMessage(content=response.content)
        ]  # 添加消息歷史
    }

# =========================
# 建立 Graph
# =========================
g = StateGraph(RAGState)
g.add_node("Retrieve", retrieve)
g.add_node("Answer", answer_with_groq)
g.add_edge(START, "Retrieve")
g.add_edge("Retrieve", "Answer")
g.add_edge("Answer", END)
app = g.compile()

# =========================
# 互動式提問迴圈 - 使用正確的狀態管理
# =========================
def repl():
    print("==== 法律條文 RAG（Chroma + LangGraph）— 輸入 exit 離開 ====")
    print(f"已索引文件：{', '.join(sorted({d.metadata['source'] for d in DOCS}))}")
    
    # 為了示範對話歷史，我們可以在這裡維護一個持久的狀態
    conversation_history = []
    
    while True:
        try:
            q = input("\n問題> ").strip()
            if q.lower() == "exit":
                print("Bye!")
                break
            if not q:
                continue
            
            # 構建初始狀態 - 包含所有必要字段
            initial_state: RAGState = {
                "question": q,
                "context": "",
                "answer": "",
                "retrieved_docs": [],
                "messages": conversation_history.copy()  # 保持對話歷史
            }
            
            # 執行 LangGraph - 狀態會在節點間正確傳遞
            result = app.invoke(initial_state)
            
            # 顯示結果
            print("\n[檢索到的參考內容]")
            print(result.get("context", "(無)"))
            print("\n[回答]")
            print(result.get("answer", "(無)"))
            
            # 更新對話歷史
            conversation_history.extend([
                HumanMessage(content=q),
                AIMessage(content=result.get("answer", ""))
            ])
            
            # 顯示狀態追蹤資訊（除錯用）
            print(f"\n[Debug] 檢索文檔數：{len(result.get('retrieved_docs', []))}")
            print(f"[Debug] 對話歷史長度：{len(result.get('messages', []))}")
            
        except KeyboardInterrupt:
            print("\n(按 Ctrl+C 離開或輸入 exit)")
        except Exception as e:
            print(f"\n[錯誤] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    repl()