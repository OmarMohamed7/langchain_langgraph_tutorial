from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

class Review(TypedDict):
    summary: str
    sentiment: str

structured_llm = llm.with_structured_output(Review)

result = structured_llm.invoke("The product is great!")

print(result)