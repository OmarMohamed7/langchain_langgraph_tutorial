from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that can answer questions about {topic}."),
    ("user", "{input}"),
])

prompt = chat_template.invoke({
    "topic": "cars",
    "input": "What is the best car in the world?"
})

result = llm.invoke(prompt)

print(result.content)