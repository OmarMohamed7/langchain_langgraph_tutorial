from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(model="phi4-mini:latest")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


lllm_with_tools = model.bind_tools([add])

res = lllm_with_tools.invoke("What is 2 + 3?")
print(res)
