from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional
import os

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import get_runtime

# 加载 .env 文件
load_dotenv()

# ------------------------------
# 定义系统提示
# ------------------------------
system_prompt = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location.
If you can tell from the question that they mean wherever they are,
use the get_user_location tool to find their location.
"""

# ------------------------------
# 定义上下文 Schema
# ------------------------------
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# ------------------------------
# 定义工具
# ------------------------------
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_user_location() -> str:
    """Retrieve user location based on user ID."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "San Francisco"

# ------------------------------
# 配置 LLM
# ------------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)


# ------------------------------
# 定义响应格式
# ------------------------------
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: Optional[str] = None

# ------------------------------
# 设置记忆存储
# ------------------------------
checkpointer = InMemorySaver()

# ------------------------------
# 创建 Agent
# ------------------------------
agent = create_agent(
    model=llm,
    prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer
)

# ------------------------------
# 运行 Agent
# ------------------------------
config = {"configurable": {"thread_id": "1"}}

# 第一次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print("第一次对话结果：")
print(response['messages'][-1].content)

# 第二次对话（同一会话 thread_id，能保持记忆）
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print("第二次对话结果：")
print(response['messages'][-1].content)