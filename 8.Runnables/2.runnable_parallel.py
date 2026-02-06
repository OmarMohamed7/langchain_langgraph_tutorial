from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(model="phi4-mini:latest")

prompt1 = PromptTemplate(
    template="generate a tweet about {topic}?",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a linkedin post about {topic}?",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "tweet": prompt1
        | model
        | parser,  # equivelent to RunnableSequence(prompt1, model, parser)
        "linkedin_post": prompt2
        | model
        | parser,  # equivelent to RunnableSequence(prompt2, model, parser)
    }
)

result = chain.invoke({"topic": "Python"})

print(result)
