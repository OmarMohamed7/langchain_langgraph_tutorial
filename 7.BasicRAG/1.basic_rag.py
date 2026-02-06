from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# Step1 : load env variables
load_dotenv()

# Step2 : load the documents
loader = TextLoader("docs.txt")
documents = loader.load()

# Step3 : split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Step4 : create the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step5 : create the retrieval pipeline
retriever = vectorstore.as_retriever()

# Step6: Initialize the model and prompt
model = ChatOllama(model="phi4-mini:latest")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question "{input}" based solely on the context below:
<context>
{context}
</context>
If you can't find an answer, say "I don't know." 
"""
)

# Step7: Create the two chains
#
# Why two chains?
# 1. document_chain: Takes {context, input} → generates answer using LLM
# 2. retrieval_chain: Takes {input} → finds relevant docs → passes to document_chain → returns answer
#
# Flow: Question → retrieval_chain → retriever finds docs → document_chain → LLM → Answer

document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# Step8: Query the chain
query = (
    "what are the key takeaways from the document?"  # Replace with your actual question
)
response = retrieval_chain.invoke({"input": query})

print(response)
