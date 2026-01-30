# Free LLM Options for LangChain

Here are **completely free** alternatives to Google Gemini API:

## 🏆 **Best Options (Ranked)**

### 1. **Ollama** ⭐ RECOMMENDED
- **Cost**: 100% FREE
- **Setup**: Runs locally on your machine
- **Speed**: Fast (depends on your hardware)
- **Privacy**: Complete privacy (runs offline)
- **Installation**:
  ```bash
  # Install Ollama
  brew install ollama  # macOS
  # or download from https://ollama.ai
  
  # Install Python package
  pip install langchain-ollama
  
  # Download a model
  ollama pull llama3.2
  # or: ollama pull mistral
  # or: ollama pull qwen2.5
  ```
- **Usage**: See `parallel_chain_ollama.py`
- **Pros**: No API keys, completely free, works offline
- **Cons**: Requires local installation, uses your computer's resources

### 2. **Groq** ⚡ FASTEST
- **Cost**: FREE tier (very generous limits)
- **Setup**: Just need free API key
- **Speed**: Extremely fast (uses specialized hardware)
- **Installation**:
  ```bash
  pip install langchain-groq
  ```
- **Get API Key**: https://console.groq.com/ (free signup)
- **Usage**: See `parallel_chain_groq.py`
- **Pros**: Very fast, easy setup, generous free tier
- **Cons**: Requires internet, API key needed

### 3. **Hugging Face** 🤗
- **Cost**: FREE (with free API token)
- **Setup**: Get free token from Hugging Face
- **Speed**: Moderate (depends on model)
- **Installation**:
  ```bash
  pip install langchain-huggingface transformers
  ```
- **Get API Token**: https://huggingface.co/settings/tokens (free)
- **Usage**: See `parallel_chain_huggingface.py`
- **Pros**: Many free models, good for experimentation
- **Cons**: Some rate limits, requires API token

## 📊 **Quick Comparison**

| Feature | Ollama | Groq | Hugging Face |
|---------|--------|------|--------------|
| **Cost** | Free | Free tier | Free |
| **Setup** | Medium | Easy | Easy |
| **Speed** | Fast | Very Fast | Moderate |
| **Privacy** | 100% Private | Cloud | Cloud |
| **Offline** | Yes | No | No |
| **API Key** | No | Yes | Yes |

## 🚀 **Quick Start (Ollama - Recommended)**

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Install package: `pip install langchain-ollama`
4. Use `parallel_chain_ollama.py`

## 📝 **Code Examples**

All three alternatives use the same chain structure - just change the LLM initialization:

```python
# Ollama
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

# Groq
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key="...")

# Hugging Face
from langchain_huggingface import ChatHuggingFace
llm = ChatHuggingFace(model_id="mistralai/Mistral-7B-Instruct-v0.2")
```
