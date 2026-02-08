from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = ChatOllama(model="phi4-mini:latest")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# step1 : indexing ( document ingestion)
video_id = "DlIAd4Rtkr8"

try:
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id)
    transcript = " ".join([i.text for i in transcript_list])

except Exception as e:
    print(f"Error fetching transcript: {e}")
    exit(1)

# Step2 : Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents([transcript])

# Step3 : Creating Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step4 : Retriver
retriver = vectorstore.as_retriever(
    search_kwargs={"k": 3},
    search_type="similarity",
)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant that can answer questions
    answer only from provided transcript, do not make up any information.
    if the context is insufficent, say "I don't know."
    context: {context}
    question: {question}
    answer:
    """,
    input_variables=["context", "question"],
)

question = "what does the creeator say about brazil in the world cup football 2026?"

retrived_docs = retriver.invoke(question)

context_text = "\n\n".join([doc.page_content for doc in retrived_docs])

final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)

print(answer.content)
