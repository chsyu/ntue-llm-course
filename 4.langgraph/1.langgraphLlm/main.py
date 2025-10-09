# main.py
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# 狀態 schema：只需 question 與 answer
class QAState(TypedDict, total=False):
    question: str
    answer: str

# 節點：呼叫 Groq LLM 產生回答
def answer_with_groq(state: QAState) -> Dict[str, Any]:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # 可換 gemma2-9b-it 或 mixtral-8x7b
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    q = state.get("question", "")
    out = llm.invoke(f"請用簡潔繁體中文回答下列問題：{q}")
    return {"answer": out.content}

# 建立 Graph
g = StateGraph(QAState)
g.add_node("AnswerGroq", answer_with_groq)
g.add_edge(START, "AnswerGroq")
g.add_edge("AnswerGroq", END)
app = g.compile()

# 主程式：可互動式輸入問題
if __name__ == "__main__":
    print("=== LangGraph 簡化版問答系統（輸入 exit 離開）===")
    while True:
        q = input("\n問題> ").strip()
        if q.lower() == "exit":
            print("再見 👋")
            break
        if not q:
            continue

        state: QAState = {"question": q}
        result = app.invoke(state)

        print("\n--- 回答 ---")
        print(result.get("answer", "(無回答)"))