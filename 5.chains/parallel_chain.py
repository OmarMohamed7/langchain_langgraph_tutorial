from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama


load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
# )

llm = ChatOllama(model="phi4-mini:latest")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes for the following topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answer for the following topic: {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="merge the provided notes and quiz into a single document \n notes: {notes} \n quiz: {quiz}",
    input_variables=["notes", "quiz"],
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | llm | parser,
        "quiz": prompt2 | llm | parser,
    }
)

sequential_chain = prompt3 | llm | parser

chain = parallel_chain | sequential_chain

result = chain.invoke({"topic": "Python"})

print(result)
