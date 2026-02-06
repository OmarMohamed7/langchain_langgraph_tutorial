"""
RunnableSequence: Chains multiple operations sequentially
Each step processes the output of the previous step.

Flow:
1. prompt1 formats input → "write a joke about Python?"
2. model generates joke → "Why do Python programmers..."
3. parser extracts text → "Why do Python programmers..."
4. prompt2 formats joke → "Explain the joke... Why do Python..."
5. model explains joke → "This joke plays on..."
6. parser extracts final text → Final explanation

Alternative: You can use pipe operator instead:
chain = prompt1 | model | parser | prompt2 | model | parser
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOllama(model="phi4-mini:latest")

prompt1 = PromptTemplate(
    template="write a joke about {topic}?",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Explain the joke in a way that is easy to understand {joke}?",
    input_variables=["joke"],
)

parser = StrOutputParser()

# RunnableSequence chains operations: each step runs after the previous completes
# Equivalent to: prompt1 | model | parser | prompt2 | model | parser
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "Python"})

print(result)
