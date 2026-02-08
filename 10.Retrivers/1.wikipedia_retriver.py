from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "he UEFA Champions League (UCL)"

docs = retriever.invoke(query)

for doc in docs:
    print(doc)
    print("-" * 100)
