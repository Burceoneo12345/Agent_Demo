import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
import getpass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def chatbot(state: State, llm: ChatOpenAI):
    return {"messages": [llm.invoke(state["messages"])]}



if __name__ == "__main__":
    load_dotenv(find_dotenv())
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0
    )
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot, llm=llm)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()