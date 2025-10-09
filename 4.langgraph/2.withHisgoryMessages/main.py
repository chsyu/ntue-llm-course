# chat_history_trimmed.py
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages

load_dotenv()

# === 單一節點：LLM 回覆 + 自動 trim ===
def chat_llm(state: MessagesState):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )

    # 用 trimmed 給 LLM，而不是整個 state["messages"]
    response = llm.invoke(state["messages"])

    # 回傳「局部更新」的訊息清單
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