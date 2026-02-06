# Quick Guide: Swapping Vector Stores (No Code Changes!)

All vector stores in LangChain use the **same interface**, so you can swap them with **minimal changes**.

## 🔄 **How to Switch (3 Steps)**

### **Current Code (FAISS):**
```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### **To Switch to Another Store:**

1. **Change the import** (line 8)
2. **Change the vectorstore creation** (line 25)
3. **Everything else stays the same!** ✨

---

## 📋 **Quick Reference: All Alternatives**

### **1. Chroma** (Easiest for Development)
```python
# Install: pip install chromadb
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(chunks, embeddings)
# Optional: Add persistence
# vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
```

### **2. FAISS** (Current - Fast, Local)
```python
# Already installed: faiss-cpu
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
# Optional: Save to disk
# vectorstore.save_local("./faiss_index")
```

### **3. In-Memory** (Testing Only)
```python
# No installation needed
from langchain_community.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
```

### **4. Qdrant** (Open Source, Fast)
```python
# Install: pip install qdrant-client
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    chunks, 
    embeddings,
    location=":memory:"  # or "http://localhost:6333" for server
)
```

### **5. Pinecone** (Cloud, Production)
```python
# Install: pip install pinecone-client
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    chunks, 
    embeddings,
    index_name="my-index"
)
```

### **6. Weaviate** (GraphQL Interface)
```python
# Install: pip install weaviate-client
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore.from_documents(chunks, embeddings)
```

### **7. Milvus** (Scalable, Production)
```python
# Install: pip install pymilvus
from langchain_milvus import MilvusVectorStore

vectorstore = MilvusVectorStore.from_documents(chunks, embeddings)
```

---

## 🎯 **Which One Should You Use?**

| Use Case | Recommended |
|----------|-------------|
| **Learning/Testing** | Chroma or In-Memory |
| **Local Development** | FAISS or Chroma |
| **Production (Small)** | Chroma or Qdrant |
| **Production (Large)** | Pinecone, Milvus, or Qdrant |
| **Need GraphQL** | Weaviate |
| **Maximum Speed** | FAISS or Qdrant |

---

## ⚡ **Example: Switching to Chroma**

**Before (FAISS):**
```python
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
```

**After (Chroma):**
```python
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(chunks, embeddings)
```

**That's it!** The rest of your code (retriever, chains, etc.) works exactly the same.

---

## 🔍 **Similarity Metrics**

Most vector stores support multiple similarity metrics:

- **Cosine Similarity**: Best for text embeddings (most common)
- **Euclidean Distance (L2)**: Also good for embeddings
- **Inner Product (Dot Product)**: When vectors are normalized

**FAISS Default**: L2 (Euclidean)
**Chroma Default**: Cosine
**Most Others**: Cosine

You can usually change this when creating the vectorstore, but defaults work well for most cases!
