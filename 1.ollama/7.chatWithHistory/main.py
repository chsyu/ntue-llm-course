from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, trim_messages
import asyncio

LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。嚴禁生成不符合事實的內容。"

# 全局聊天歷史存儲（可換成 Redis/DB）
chat_histories: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """取得/建立該 session 的歷史，並在回傳前做截斷控制。"""
    h = chat_histories.get(session_id)
    if h is None:
        h = InMemoryChatMessageHistory()
        chat_histories[session_id] = h
    # 使用 LangChain 原生的 trim_messages 進行截斷
    if h.messages:
        h.messages = trim_messages(h.messages, max_tokens=24, token_counter=len, include_system=True)
    return h

def create_chat_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_history


async def chat_with_history(chain, user_input: str, session_id: str = "default"):
    config = {"configurable": {"session_id": session_id}}
    return await chain.ainvoke({"input": user_input}, config=config)

def print_welcome():
    print("=" * 50)
    print("🤖 LangChain 聊天機器人")
    print("=" * 50)
    print("輸入 'quit' 或 'exit' 結束對話")
    print("輸入 'clear' 清除聊天歷史")
    print("輸入 'history' 查看聊天歷史")
    print("-" * 50)

_ROLE_MAP = {
    "system": "系統",
    "human": "用戶",
    "ai": "助手",
    "tool": "工具",
    "function": "函式",
}

def print_chat_history(session_id: str = "default"):
    h = chat_histories.get(session_id)
    if not h or not h.messages:
        print("目前沒有聊天歷史。")
        return
    print("\n📋 聊天歷史:")
    print("-" * 30)
    for i, m in enumerate(h.messages, 1):
        role = _ROLE_MAP.get(m.type, m.type)
        print(f"{i}. {role}: {getattr(m, 'content', '')}")
    print("-" * 30)

def clear_chat_history(session_id: str = "default"):
    chat_histories[session_id] = InMemoryChatMessageHistory()
    print("✅ 聊天歷史已清除。")

async def main():
    print_welcome()
    chain = create_chat_chain()
    session_id = "default"
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            low = user_input.lower()
            if low in {"quit", "exit", "退出"}:
                print("\n👋 再見！")
                break
            if low in {"clear", "清除"}:
                clear_chat_history(session_id); continue
            if low in {"history", "歷史"}:
                print_chat_history(session_id); continue
            if not user_input:
                print("請輸入一些內容..."); continue

            print("🤔 正在思考...")
            resp = await chat_with_history(chain, user_input, session_id)
            print(f"\n🤖 助手: {resp}")

        except KeyboardInterrupt:
            print("\n\n👋 收到中斷訊號，再見！")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            print("請重試...")

if __name__ == "__main__":
    asyncio.run(main())