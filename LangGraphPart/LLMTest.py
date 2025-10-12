from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)



response = llm.invoke("你好，请介绍一下自己")
print(response.content)