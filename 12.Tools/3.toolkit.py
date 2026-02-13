from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers together"""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


class MathTollKit:
    def get_tools(self):
        return [add, subtract, multiply]


math_toolkit = MathTollKit()

tools = math_toolkit.get_tools()

for tool in tools:
    print(tool.name)
    print(tool.description)
    print(tool.invoke({"a": 2, "b": 3}))
    print("-" * 100)
