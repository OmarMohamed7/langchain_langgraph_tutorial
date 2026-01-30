from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt1 = PromptTemplate(template="Hello, what is the capital of {country}?", input_variables=["country"])
prompt2 = PromptTemplate(template="What is the population of {country}?", input_variables=["country"])
prompt3 = PromptTemplate(template="What is the area of {country}?", input_variables=["country"])

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser | prompt3 | llm | parser

result = chain.invoke({"country": "Germany"})

print(result)
