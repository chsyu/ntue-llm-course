# chat_history_trimmed.py
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages

load_dotenv()

# === 單一節點：LLM 回覆 + 簡單 trim ===
def chat_llm(state: MessagesState):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )

    messages = state["messages"]
    
    # 簡單做法：只保留最近的 20 條消息（約 10 輪對話）
    if len(messages) > 20:
        messages = messages[-20:]

    response = llm.invoke(messages)
    return {"messages": [response]}

# === 建立 LangGraph ===
graph = StateGraph(MessagesState)
graph.add_node("Chat", chat_llm)
graph.add_edge(START, "Chat")
graph.add_edge("Chat", END)

# === 啟用記憶體型 checkpointer ===
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# === 互動迴圈 ===
if __name__ == "__main__":
    print("=== LangGraph + Chat History + Trim ===")
    thread_id = "demo-thread-1"

    while True:
        q = input("\n你> ").strip()
        if q.lower() == "exit":
            print("掰掰～")
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        print("AI>", result["messages"][-1].content)