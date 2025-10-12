from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
import asyncio

LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。"

# 儲存多個 session 的對話歷史
chat_histories: dict[str, list] = {}

def get_session_messages(session_id: str) -> list:
    """取得或建立該 session 的對話歷史"""
    if session_id not in chat_histories:
        chat_histories[session_id] = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT)]
    return chat_histories[session_id]

# --- Step 1: 建立 RunnableLambda，用來生成 messages ---
def build_messages(input_dict: dict):
    """組裝完整 messages 給 LLM"""
    session_id = input_dict["session_id"]
    user_input = input_dict["input"]
    messages = get_session_messages(session_id)

    # 加入 HumanMessage
    messages.append(HumanMessage(content=user_input))
    return messages

# --- Step 2: 定義 LLM 與輸出解析 ---
llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
parser = StrOutputParser()

# --- Step 3: 用 LCEL 串接 ---
chain = (
    RunnableLambda(build_messages)
    | llm
    | parser
)

# --- Step 4: 執行函數 ---
async def chat_with_history(user_input: str, session_id: str = "default"):
    # 傳入 input + session_id，讓 build_messages 有資訊組合 messages
    result = await chain.ainvoke({"input": user_input, "session_id": session_id})

    # 執行完後 append AI 回覆
    messages = get_session_messages(session_id)
    messages.append(AIMessage(content=result))
    return result

def print_welcome():
    print("=" * 50)
    print("🤖 LangChain 聊天機器人")
    print("=" * 50)

async def main():
    print_welcome()
    session_id = "default"

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in {"quit", "exit", "退出"}:
            print("\n👋 再見！")
            break

        print("🤔 正在思考...")
        resp = await chat_with_history(user_input, session_id)
        print(f"\n🤖 助手: {resp}")

if __name__ == "__main__":
    asyncio.run(main())