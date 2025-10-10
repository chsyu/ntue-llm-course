"""
簡潔的 LangGraph 天氣助手 - 修正一週天氣功能
"""
from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from weather_tools import get_current_weather, get_weather_forecast

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 更自然的系統提示
SYSTEM_MESSAGE = SystemMessage(content="""你是專業的天氣助手，用繁體中文回答。

你有兩個工具可以使用：
- get_current_weather(city): 查詢指定城市的當前天氣狀況
- get_weather_forecast(city, days): 查詢指定城市的未來天氣預報

請根據用戶問題的自然語意智能選擇合適的工具：
- 如果用戶想知道現在的天氣情況，使用當前天氣工具
- 如果用戶想知道未來的天氣預報，使用預報工具
- 一週預報請設定 days=7

相信你的語言理解能力，根據問題的語意做出最佳判斷。""")

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
tools = [get_current_weather, get_weather_forecast]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State) -> dict:
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SYSTEM_MESSAGE] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    return "continue" if hasattr(last_message, 'tool_calls') and last_message.tool_calls else "end"

# 構建圖形
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=MemorySaver())

async def chat(query: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    try:
        result = await app.ainvoke({"messages": [HumanMessage(content=query)]}, config)
        return result["messages"][-1].content
    except Exception as e:
        return f"錯誤: {str(e)}"