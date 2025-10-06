from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional
import os
import json
import re

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

IMPORTANT: After getting the weather information, you MUST respond in valid JSON format like this:
{
  "punny_response": "Your weather forecast with puns",
  "weather_conditions": "The actual weather conditions or null"
}

Make sure your response is ONLY the JSON object, nothing else.
"""

# ------------------------------
# 定义上下文 Schema
# ------------------------------
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# ------------------------------
# 定义响应格式
# ------------------------------
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: Optional[str] = None

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
# 解析响应格式
# ------------------------------
def parse_response(content: str) -> ResponseFormat:
    """从字符串中解析 ResponseFormat"""
    try:
        # 尝试直接解析 JSON
        # 提取 JSON 部分（可能包含在其他文本中）
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            return ResponseFormat(
                punny_response=data.get("punny_response", ""),
                weather_conditions=data.get("weather_conditions")
            )
    except json.JSONDecodeError:
        pass
    
    # 如果不是 JSON，尝试解析原格式（- punny_response: xxx）
    punny_match = re.search(r'-\s*punny_response:\s*(.+?)(?=\n-|$)', content, re.DOTALL)
    weather_match = re.search(r'-\s*weather_conditions:\s*(.+?)(?=\n-|$)', content, re.DOTALL)
    
    if punny_match:
        punny_response = punny_match.group(1).strip()
        weather_conditions = weather_match.group(1).strip() if weather_match else None
        return ResponseFormat(
            punny_response=punny_response,
            weather_conditions=weather_conditions
        )
    
    # 如果都失败，返回原始内容
    return ResponseFormat(punny_response=content, weather_conditions=None)

# ------------------------------
# 运行 Agent
# ------------------------------
config = {"configurable": {"thread_id": "1"}}

# 第一次对话
print("=" * 50)
print("第一次对话：")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

content = response['messages'][-1].content
print(f"原始回复:{content}")

# 解析为结构化格式
parsed = parse_response(content)
print("结构化响应：")
print(f"双关回复:{parsed.punny_response}")
print(f"天气状况:{parsed.weather_conditions}")

# 第二次对话（同一会话 thread_id，能保持记忆）
print("\n" + "=" * 50)
print("第二次对话：")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(f"原始回复: {response['messages'][-1].content}")