"""
LangGraph Studio 應用程式
簡單的聊天機器人，使用 Ollama 本地模型
"""
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

LLM_MODEL = "qwen2.5:7b-instruct"


def chat_node(state: MessagesState) -> dict:
    """
    聊天節點：處理用戶訊息並產生回應
    
    Args:
        state: 包含訊息歷史的狀態
        
    Returns:
        dict: 包含新訊息的字典
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def create_graph() -> StateGraph:
    """
    建立並配置 LangGraph
    
    Returns:
        StateGraph: 配置好的圖形結構
    """
    graph = StateGraph(MessagesState)
    graph.add_node("Chat", chat_node)
    graph.add_edge(START, "Chat")
    graph.add_edge("Chat", END)
    return graph


# 建立並編譯應用
graph = create_graph()
app = graph.compile()