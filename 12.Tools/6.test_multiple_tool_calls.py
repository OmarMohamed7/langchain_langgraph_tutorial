from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(model="mistral:latest")


@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """Return the real time currency conversion rate between two currencies"""
    conversion_rate = 140.3
    return conversion_rate


# conversion_rate is float but it should be injected automatically by the system not the LLM.
@tool
def convert(
    base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Convert a value from one currency to another"""
    return base_currency_value * conversion_rate


llm_with_tools = llm.bind_tools([get_conversion_rate, convert])

user_query = "what is the conversion rate between USD and EGP and based on it, convert 100 USD to EGP"

messages = [HumanMessage(content=user_query)]

ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)

print(ai_msg)
while ai_msg.content == "" and ai_msg.tool_calls:
    tool_messages = []

    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_id = tool_call["id"]
        tool_args = tool_call["args"]

        print(f"Tool name: {tool_name}")
        print(f"Tool id: {tool_id}")
        print(f"Tool args: {tool_args}")
        print("-" * 100)

        if tool_name == "get_conversion_rate":
            rate = get_conversion_rate.invoke(tool_args)
            tool_output = str(rate)
            conversion_rate = rate

        elif tool_name == "convert":
            if "conversion_rate" not in tool_args:
                if "conversion_rate" not in locals() or conversion_rate is None:
                    raise ValueError("Conversion rate is required")
                else:
                    tool_args["conversion_rate"] = conversion_rate

            result = convert.invoke(tool_args)
            tool_output = str(result)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_msg = ToolMessage(content=tool_output, tool_call_id=tool_id)
        tool_messages.append(tool_msg)

    messages.append(ai_msg)
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

print(ai_msg.content)
