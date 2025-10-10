# 簡化 RAG with LangGraph
from dotenv import load_dotenv
import os
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_community.vectorstores import Chroma

load_dotenv()

# 狀態累積函數
def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    return left + right

# === 簡化狀態定義 ===
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    context: str

# === 載入向量資料庫 ===
def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

VECTORSTORE = get_vectorstore()

# === 節點 1: 檢索 ===
def retrieve(state: RAGState) -> RAGState:
    question = state.get("question", "")
    docs = VECTORSTORE.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {
        **state,
        "context": context
    }

# === 節點 2: 回答（包含對話歷史）===
def answer_with_context(state: RAGState) -> RAGState:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )
    
    question = state.get("question", "")
    context = state.get("context", "")
    messages = state.get("messages", [])
    
    # 系統提示
    system_prompt = f"""請根據以下參考資料回答問題，用繁體中文回答：

參考資料：
{context}"""
    
    # 簡單記憶管理：保留最近 6 條消息
    if len(messages) > 6:
        messages = messages[-6:]
    
    # 構建完整對話（包含歷史）
    conversation = [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=question)]
    
    response = llm.invoke(conversation)
    
    return {
        **state,
        "messages": [AIMessage(content=response.content)]
    }

# === 建立 LangGraph ===
graph = StateGraph(RAGState)
graph.add_node("Retrieve", retrieve)
graph.add_node("Answer", answer_with_context)

graph.add_edge(START, "Retrieve")
graph.add_edge("Retrieve", "Answer") 
graph.add_edge("Answer", END)

# === 啟用自動記憶管理 ===
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# === 互動迴圈 ===
if __name__ == "__main__":
    print("=== 簡化 RAG 系統（輸入 exit 離開）===")
    
    thread_id = "rag-chat"
    
    while True:
        q = input("\n問題> ").strip()
        if q.lower() == "exit":
            print("再見！")
            break
        if not q:
            continue
        
        # 簡單狀態
        initial_state: RAGState = {
            "messages": [],
            "question": q,
            "context": ""
        }
        
        # 執行 RAG（自動記憶管理）
        result = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        print(f"\n[回答] {result['messages'][-1].content}")
        print(f"[檢索] 找到 {len(result.get('context', '').split('\\n\\n'))} 個相關片段")