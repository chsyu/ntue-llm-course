import json
import os
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio

# ===============================
# 基本設定
# ===============================
LLM_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SYSTEM_PROMPT = "你是精煉且忠實的助教，禁止臆測。"

# 歷史紀錄資料夾
HISTORY_DIR = "chat_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)


# ===============================
# JSON 持久化模組
# ===============================
def get_history_path(session_id: str) -> str:
    """取得對應 session 的檔案路徑"""
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def load_history(session_id: str) -> list:
    """讀取歷史紀錄，若無檔案則建立新檔"""
    path = get_history_path(session_id)
    if not os.path.exists(path):
        # 若無檔案，建立含 system prompt 的初始歷史
        init = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT)]
        save_history(session_id, init)
        return init

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 將 dict 轉為 Message 物件
    messages = []
    for m in data:
        role = m["type"]
        if role == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif role == "human":
            messages.append(HumanMessage(content=m["content"]))
        elif role == "ai":
            messages.append(AIMessage(content=m["content"]))
    return messages

def save_history(session_id: str, messages: list):
    """將 messages 儲存成 JSON"""
    path = get_history_path(session_id)
    data = []
    for m in messages:
        data.append({
            "type": m.type,
            "content": m.content,
            "timestamp": datetime.now().isoformat()
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===============================
# 聊天邏輯
# ===============================
async def chat_with_persist(session_id: str = "default"):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

    messages = load_history(session_id)
    print("=" * 60)
    print(f"🗂️ 載入 session: {session_id}")
    print(f"📜 歷史訊息數量: {len(messages)}")
    print("=" * 60)

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in {"quit", "exit", "退出"}:
            print("\n💾 儲存中...")
            save_history(session_id, messages)
            print(f"✅ 已儲存於 {get_history_path(session_id)}")
            break

        # 加入人類訊息
        messages.append(HumanMessage(content=user_input))

        # 呼叫 LLM
        print("🤔 助手思考中...")
        resp = await llm.ainvoke(messages)

        # 顯示並加入 AI 訊息
        print(f"\n🤖 助手: {resp.content}")
        messages.append(AIMessage(content=resp.content))

        # 實時儲存
        save_history(session_id, messages)


# ===============================
# 主程式執行
# ===============================
if __name__ == "__main__":
    asyncio.run(chat_with_persist("default"))