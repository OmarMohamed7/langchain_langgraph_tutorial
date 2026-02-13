from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers together"""
    return a - b


result = multiply.invoke({"a": 2, "b": 3})
print(result)
