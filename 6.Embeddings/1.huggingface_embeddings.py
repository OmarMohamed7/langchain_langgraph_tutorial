from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Welcome, how are you?"

documents = ["Hello, how are you?", "I am fine, thank you!", "How are you doing?"]

text_embedding = embeddings.embed_query(text)

documents_embedding = embeddings.embed_documents(documents)

similarity = cosine_similarity([text_embedding], documents_embedding)

print(similarity)
