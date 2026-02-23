# LangChain & LangGraph Tutorial

My hands-on tutorial covering **LangChain** and **LangGraph** for building LLM applications: chatbots, RAG (Retrieval-Augmented Generation), tools, chains, and multi-step agents.

---

## What I Learned

1. **LLM basics** – Calling models (Ollama, Google Gemini), prompts, and structured outputs  
2. **Chatbots** – Multi-turn conversation with system/user/assistant messages  
3. **Chains** – Sequential, parallel, and conditional pipelines  
4. **RAG** – Load documents, chunk, embed, store in a vector DB, and answer from them  
5. **Embeddings & vector stores** – Hugging Face embeddings, FAISS, Chroma  
6. **Retrievers** – Wikipedia and vector-store retrievers  
7. **RAG systems** – Basic RAG and YouTube transcript RAG  
8. **Tools** – Defining tools, binding to the LLM, and tool calling  
9. **LangGraph** – Graphs, state, conditional edges, and RAG agents with retrieve-then-answer  

---

## Project Structure

```
langchain-tut/
├── 1-LLMInteraction/          # First steps with LLMs (e.g. Gemini)
├── 2-Chatbots/                # Chat with system/user/AI messages
├── 3.StructuredOutput/        # Structured outputs (JSON, Pydantic)
├── 5.chains/                  # Sequential, parallel, conditional chains
├── 6.Embeddings/              # Hugging Face embeddings
├── 7.BasicRAG/                # Basic RAG: load → chunk → embed → retrieve → answer
├── 8.Runnables/               # Runnable sequence and parallel
├── 9.VectorStore/             # Chroma and FAISS vector stores
├── 10.Retrivers/              # Wikipedia and vector-store retrievers
├── 11.RAGSystems/             # RAG on YouTube transcripts
├── 12.Tools/                  # Tools, toolkits, binding, multi-tool calls
├── 13.LangGraph/              # Graphs, state, conditions, tools, RAG agent
├── requirements.txt
└── README.md
```

---

## 1. LLM Interaction

I learned to call an LLM (e.g. **Google Gemini** via `langchain_google_genai`), send prompts, and read text responses. I use `dotenv` for API keys.

**Takeaway:** LLMs are invoked with a prompt; the response is unstructured text unless I use parsing or structured output.

---

## 2. Chatbots

I use **messages:** `SystemMessage`, `HumanMessage`, `AIMessage` for multi-turn chat. I keep a list of messages and append each user turn and model reply. The system message sets behavior (e.g. “You are a helpful assistant…”).

**Takeaway:** Chat is a list of messages; the model uses full history to generate the next reply.

---

## 3. Structured Output

I learned to get **JSON** or **Pydantic**-shaped output instead of free text, using `with_structured_output()` or output parsers so the model returns a fixed schema.

**Takeaway:** I can make the LLM return data structures my code can use directly.

---

## 4. Chains

I built **sequential** chains (`prompt → llm → parser → prompt → llm → parser`), **parallel** chains (run several and combine results), and **conditional** chains (choose the next step from the model’s output). I use the **pipe** operator: `prompt | llm | parser`.

**Takeaway:** Complex flows are built by composing prompts, LLM, and parsers into chains.

---

## 5. Embeddings

I learned that **embeddings** turn text into vectors for similarity search. I use **Hugging Face** (`sentence-transformers/all-MiniLM-L6-v2`) to embed documents and queries.

**Takeaway:** Similar meaning → similar vectors; this is the basis of retrieval in RAG.

---

## 6. Basic RAG

I implemented RAG by **loading** documents (e.g. `TextLoader`), **splitting** into chunks (`RecursiveCharacterTextSplitter`: chunk size, overlap), **embedding** chunks and **storing** them in a vector store (e.g. FAISS). For each user question I **retrieve** top‑k similar chunks and **generate** an answer by passing question + retrieved chunks to the LLM.

**Takeaway:** RAG = retrieve relevant chunks from my data, then answer using only (or mainly) that context.

---

## 7. Runnables

I used **RunnableSequence** (steps run one after another) and **RunnableParallel** (run multiple runnables and merge results). Same idea as chains but with LangChain’s runnable interface and `|` for composition.

**Takeaway:** LangChain models, prompts, and parsers are “runnables” I can sequence or run in parallel.

---

## 8. Vector Stores

I used **Chroma** and **FAISS** to store embeddings and do similarity search. I index with `from_documents(documents, embeddings)` and search with `as_retriever()`.

**Takeaway:** Vector stores are the “memory” of RAG: I query them with the user question’s embedding to get relevant passages.

---

## 9. Retrievers

I built retrievers for **Wikipedia** (retrieve by query) and for my **vector store** (e.g. `vectorstore.as_retriever(search_kwargs={"k": 3})`).

**Takeaway:** A retriever is the component that, given a query, returns a list of relevant documents/chunks.

---

## 10. RAG Systems

I built **basic RAG** (text/PDF → chunks → vector store → retrieve → LLM) and **YouTube RAG** (transcript → chunks → vector store → retrieve → LLM) so I can ask questions about a video.

**Takeaway:** RAG is a pattern I can apply to any source (files, web, transcripts) once I have chunks and a retriever.

---

## 11. Tools

I learned to **define tools** with `@tool` and type hints (the docstring describes the tool for the LLM), **bind tools** to the LLM with `llm.bind_tools(tools)`, and **execute** the chosen tool with the model’s arguments and return the result (e.g. as `ToolMessage`). I also handled **multi-tool calls** (the model requests several tools in one turn; I execute them and pass results back).

**Takeaway:** Tools let the LLM trigger real actions (search, compute, APIs); I run the tool and feed the result back into the conversation.

---

## 12. LangGraph

I learned **state** (e.g. `messages: Annotated[Sequence[BaseMessage], add_messages]` with the `add_messages` reducer so new messages are appended), **graphs** with `StateGraph(State)` and nodes (functions that take state and return state updates), **edges** (`add_edge(START, "node_a")`, `add_edge("node_a", END)`, and **conditional edges** (e.g. if last message has tool calls → tools node, else → end). I **compile** with `graph.compile()` and invoke with initial state.

I built a **retrieve-then-answer RAG agent:** a **retrieve_first** node runs the retriever and injects the chunks into state; the **llm** node gets system prompt + user question + retrieved document so the answer is only from the file. I run retrieval in code (no tool-calling from the model) to avoid hallucinated tool calls and to guarantee the model sees the document (e.g. by putting it in the system message).

**Takeaway:** LangGraph is for multi-step, stateful flows (agents); I can implement a robust RAG agent by “retrieve first in code, then answer” so the model always has the file content in context.

---

## My RAG Agent (13.LangGraph/7.rag_agent.ipynb)

This notebook is my **RAG agent that searches a pre-loaded PDF and answers only from it**:

1. **Load PDF** → split into chunks → embed → store in **Chroma**.  
2. **retrieve_first:** when I ask a question, `retriever.invoke(query)` runs and the retrieved chunks are injected into state (and in my code, into the **system message** so the model always sees the file).  
3. **call_llm:** one system message (instructions + full retrieved document) + one user message (my question); the LLM answers only from that text.  
4. **No tool calls** for retrieval (avoids fake tool calls and “I can’t access files” answers).

So: **search in the pre-loaded PDF (via retriever), then answer from the file (via system message + strict prompt).**

---

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate   # or: env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Ollama for local LLMs
# Install from https://ollama.ai and run e.g. ollama pull mistral
```

I use a `.env` file for API keys (e.g. `GOOGLE_API_KEY` for Gemini) and load it with `python-dotenv` in my scripts.

---

## Main Libraries I Use

- **langchain_core** – Prompts, messages, runnables, output parsers  
- **langchain_community** – Document loaders, FAISS, etc.  
- **langchain_ollama** – Ollama chat model  
- **langchain_google_genai** – Google Gemini  
- **langchain_huggingface** – Hugging Face embeddings  
- **langchain_chroma** – Chroma vector store  
- **langchain_text_splitters** – RecursiveCharacterTextSplitter, etc.  
- **langgraph** – StateGraph, nodes, edges, conditional edges, message reducers  

---

## Summary

I learned how to:

- Call LLMs and manage chat with message lists.
- Build chains and runnables (sequential and parallel).
- Implement RAG: load → chunk → embed → store → retrieve → generate.
- Use embeddings, vector stores, and retrievers.
- Define and use tools with an LLM.
- Build a LangGraph agent with state and conditional edges.
- Build a RAG agent that **searches a pre-loaded PDF and answers only from that file** by retrieving in code and putting the document in the system message.

This gives me a solid base for building production-style RAG and agent applications with LangChain and LangGraph.
