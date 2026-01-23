from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions about the weather."),
]

while True:
    user_input = input("You: ")
    messages.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = llm.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(result.content)