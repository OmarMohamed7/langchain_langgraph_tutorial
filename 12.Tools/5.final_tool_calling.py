from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(model="phi4-mini:latest")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


llm_with_tools = llm.bind_tools([add])

user_query = "can you add 10 and 9?"

query = HumanMessage(content=user_query)

messages = [query]

response = llm_with_tools.invoke(messages)
messages.append(response)

tool_result = add.invoke(response.tool_calls[0])
messages.append(tool_result)

final_result = llm_with_tools.invoke(messages)

print(final_result.content)
