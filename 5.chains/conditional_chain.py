from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()


class Feedback(BaseModel):
    feedback: str = Field(description="The feedback for the product")
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative {feedback} and provide the response in the following format:{response_format}",
    input_variables=["feedback"],
    partial_variables={"response_format": parser2.get_format_instructions()},
)


classifier_chain = prompt1 | llm | parser2

prompt2 = PromptTemplate(
    template="Write a response to the following positivefeedback: {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write a response to the following negative feedback: {feedback}",
    input_variables=["feedback"],
)

chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | llm | parser),
    (lambda x: x.sentiment == "negative", prompt3 | llm | parser),
    RunnableLambda(lambda x: "No valid sentiment found"),
)


chain = classifier_chain | chain

result = chain.invoke({"feedback": "this is a phone"})

print(result)
