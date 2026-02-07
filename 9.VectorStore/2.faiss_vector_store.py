from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = [
    "Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    "MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    "Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    "Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
]


vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)

# FAISS returns a DISTANCE score, not a similarity score.

# So:

# 🔻 Lower score = more similar
# 🔺 Higher score = less similar

# This is the opposite of cosine similarity percentages.

# These are L2 distances in vector space.

# Score	Meaning
# ~0.7 – 0.9	Very strong semantic match
# ~1.0 – 1.2	Related but incorrect
# >1.3	Weak / irrelevant


query = "who among theses are best fast bowlers ?"
res = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in res:
    print(doc.page_content)
    print(doc.metadata)
    print(score)
    print("-" * 100)
