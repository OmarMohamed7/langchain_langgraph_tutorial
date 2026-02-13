from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    # This class considered as a schema for the input
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")


def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers together",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 2, "b": 3})
print(result)
