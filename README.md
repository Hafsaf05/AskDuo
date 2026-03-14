# 🤖 AskDuo — RAG-Powered AI Assistant
> Internal AI knowledge assistant that answers employee questions using company documents through a Retrieval-Augmented Generation (RAG) pipeline.

A modular Retrieval-Augmented Generation (RAG) assistant built with LangChain, Chroma, and OpenAI GPT-4o-mini.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?logo=openai)](https://openai.com/)
[![Chroma](https://img.shields.io/badge/VectorDB-Chroma-purple)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📖 Overview

**AskDuo** is an internal AI knowledge assistant designed to help employees quickly retrieve answers from company documentation using Retrieval-Augmented Generation (RAG).

Instead of manually searching through policies, manuals, or internal guides, employees can simply ask questions and the system retrieves the most relevant information from the organization’s knowledge base.

Unlike a raw LLM chat interface, AskDuo:
- **Reads from your knowledge base** — not just from the model's training data
- **Cites its sources** — every answer includes the document chunks used
- **Avoids hallucination** — the prompt explicitly instructs the model to answer only from retrieved context
- **Persists its vector store** — no re-embedding needed on repeat runs

This project is designed as a clean, modular, production-ready RAG template that you can extend with your own documents, embeddings, and LLMs.

---

## ✨ Key Features

• Retrieval-Augmented Generation (RAG) pipeline  
• Document chunking with overlap for semantic retrieval  
• Vector search using ChromaDB  
• OpenAI embeddings for semantic similarity  
• Context-grounded LLM responses (reduced hallucination)  
• Persistent vector store for fast repeated queries  
• Modular architecture for easy extension

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AskDuo RAG Pipeline                     │
└─────────────────────────────────────────────────────────────┘

  📄 data/documents.txt
         │
         ▼
  ┌─────────────┐
  │   Loader    │  TextLoader (LangChain)
  │  loader.py  │  → Reads raw text into Document objects
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   Chunker   │  RecursiveCharacterTextSplitter
  │  chunker.py │  → Splits docs into 500-char chunks (100 overlap)
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │  Embeddings  │  OpenAI text-embedding-3-small
  │embeddings.py │  → Converts each chunk into a 1536-dim vector
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Vector Store │  ChromaDB (persisted to ./vectorstore/)
  │vectorstore.py│  → Stores & indexes all chunk vectors
  └──────┬───────┘
         │
     Query time
         │
  ┌──────┴───────┐
  │  Retriever   │  Cosine Similarity Search (top-4 chunks)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   QA Chain   │  LangChain RetrievalQA
  │  qa_chain.py │  → Injects context into RAG prompt
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │    OpenAI    │  gpt-4o-mini
  │   GPT-4o     │  → Generates grounded, cited answer
  └──────┬───────┘
         │
         ▼
  ✅ Answer + Sources
```

---

## 📁 Project Structure

```
askduo/
│
├── app.py                  # Main CLI entrypoint
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
│
├── data/
│   └── documents.txt       # Your knowledge base (edit this!)
│
├── src/
│   ├── __init__.py         # Package exports
│   ├── loader.py           # Document loading (TextLoader)
│   ├── chunker.py          # Text splitting (RecursiveCharacterTextSplitter)
│   ├── embeddings.py       # OpenAI embeddings (text-embedding-3-small)
│   ├── vectorstore.py      # Chroma vector DB + retriever
│   └── qa_chain.py         # LangChain RetrievalQA chain (gpt-4o-mini)
│
└── vectorstore/            # Chroma persisted index (auto-created)
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/yourname/askduo.git
cd askduo
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

**5. Add your documents**

Edit `data/documents.txt` with your own knowledge base content, or use the included sample (covers AI, ML, LLMs, RAG, and vector databases).

---

## 🚀 Usage

```bash
python app.py
```

AskDuo will:
1. Load documents from `data/documents.txt`
2. Split and embed them into Chroma
3. Launch an interactive CLI session

```
🚀 Initializing AskDuo RAG Pipeline...

[Loader] ✓ Loaded 1 document(s) from 'data/documents.txt'
[Chunker] ✓ Split 1 document(s) into 42 chunk(s)
[Embeddings] ✓ Initialized OpenAI embeddings (text-embedding-3-small)
[VectorStore] ⚙  Embedding 42 chunks into Chroma...
[VectorStore] ✓ Vector store built and persisted at './vectorstore'
[VectorStore] ✓ Retriever ready (top-4 similarity search)
[QAChain] ✓ RetrievalQA chain built (gpt-4o-mini, chain_type=stuff)

✅ Pipeline ready in 8.43s

╔══════════════════════════════════════════════╗
║                                              ║
║         🤖  A S K D U O                     ║
║         RAG-Powered AI Assistant             ║
║                                              ║
║  Type your question and press Enter.         ║
║  Type  'exit'  to quit.                      ║
║  Type  'help'  for tips.                     ║
║                                              ║
╚══════════════════════════════════════════════╝

❓ Your question: How do employees apply for reimbursement?

🔍 Searching knowledge base...

💬 Answer:
Employees can submit reimbursement requests through the internal expense portal.
After submitting receipts and expense details, the request is reviewed by the finance
team and processed within 5–7 business days.

📎 Sources used:
  • [data/documents.txt] Reimbursement Policy: Employees must submit expenses...
  • [data/documents.txt] Finance Guidelines: Expense claims should be submitted...
  
⏱  Answered in 2.31s
```

---

## 🔧 Configuration

You can tune the following parameters in the source files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `chunker.py` | `500` | Max characters per chunk |
| `chunk_overlap` | `chunker.py` | `100` | Overlap between chunks |
| `k` (top results) | `vectorstore.py` | `4` | Chunks retrieved per query |
| `temperature` | `qa_chain.py` | `0.2` | LLM creativity (0=deterministic) |
| `embedding model` | `embeddings.py` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM model` | `qa_chain.py` | `gpt-4o-mini` | OpenAI chat model |

---

## 📚 Extending AskDuo

**Use your own documents:**
Replace or append to `data/documents.txt` with any text content — product docs, research papers, internal wikis, etc.

**Support PDFs:**
Replace `TextLoader` in `loader.py` with `PyPDFLoader` from `langchain_community`:
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("your_file.pdf")
```

**Load entire directories:**
Use `DirectoryLoader` to load all `.txt` or `.pdf` files at once:
```python
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("data/", glob="**/*.txt")
```

**Switch to a different LLM:**
Replace `ChatOpenAI` in `qa_chain.py` with any LangChain-supported LLM:
```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
```

---

## 🛡️ Security

- **Never commit your `.env` file.** It is listed in `.gitignore`.
- API keys are read at runtime via `python-dotenv`.
- The `vectorstore/` directory contains embeddings only — no raw text.

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| LLM Framework | LangChain 0.2 |
| Language Model | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | ChromaDB |
| Env Management | python-dotenv |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a pull request or file an issue.

---

<p align="center">Built with ❤️ using LangChain + OpenAI + Chroma</p>
