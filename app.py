"""
app.py
------
AskDuo — RAG-powered AI Assistant
Main entry point for the command-line interface.

Pipeline:
  1. Load documents from data/documents.txt
  2. Split into overlapping chunks
  3. Generate OpenAI embeddings (text-embedding-3-small)
  4. Store in Chroma vector database
  5. Build RetrievalQA chain (gpt-4o-mini)
  6. Interactive Q&A loop until user types 'exit'

Usage:
  python app.py
"""

import os
import sys
import time

from dotenv import load_dotenv

# ── Load environment variables from .env ────────────────────────────────────
load_dotenv()

# ── Add src/ to module path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.loader import load_documents
from src.chunker import split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import build_vectorstore, get_retriever
from src.qa_chain import build_qa_chain, ask


# ── Configuration ────────────────────────────────────────────────────────────
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "data", "documents.txt")
BANNER = """
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
"""

HELP_TEXT = """
💡 Tips for better answers:
  • Ask specific questions about the loaded documents.
  • Example: "What is the main purpose of X?"
  • Example: "How does Y work?"
  • Example: "Summarize the section on Z."
  • AskDuo only answers from the loaded knowledge base.
"""


def print_sources(sources: list) -> None:
    """Display the source document chunks used to generate the answer."""
    if not sources:
        return
    print("\n📎 Sources used:")
    seen = set()
    for doc in sources:
        source = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:120].replace("\n", " ").strip()
        key = snippet[:60]
        if key not in seen:
            seen.add(key)
            print(f"  • [{source}] {snippet}...")


def initialize_pipeline() -> object:
    """
    Run the full RAG pipeline initialization.

    Returns the ready-to-use QA chain.
    """
    print("\n🚀 Initializing AskDuo RAG Pipeline...\n")
    start = time.time()

    # Step 1: Load documents
    documents = load_documents(DOCUMENTS_PATH)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Initialize embedding model
    embedding_model = get_embedding_model()

    # Step 4: Build Chroma vector store
    vectorstore = build_vectorstore(chunks, embedding_model)

    # Step 5: Create retriever
    retriever = get_retriever(vectorstore)

    # Step 6: Build QA chain
    qa_chain = build_qa_chain(retriever)

    elapsed = time.time() - start
    print(f"\n✅ Pipeline ready in {elapsed:.2f}s\n")

    return qa_chain


def run_cli(qa_chain) -> None:
    """
    Run the interactive CLI question-answering loop.

    Continues prompting the user until they type 'exit'.
    """
    print(BANNER)

    while True:
        try:
            user_input = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Thanks for using AskDuo.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n👋 Goodbye! Thanks for using AskDuo.")
            break

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        # ── Generate answer ──────────────────────────────────────────────
        print("\n🔍 Searching knowledge base...\n")
        start = time.time()

        result = ask(qa_chain, user_input)

        elapsed = time.time() - start
        answer = result["answer"]
        sources = result["sources"]

        print(f"💬 Answer:\n{answer}")
        print_sources(sources)
        print(f"\n⏱  Answered in {elapsed:.2f}s")
        print("\n" + "─" * 50 + "\n")


def main() -> None:
    """Main application entrypoint."""
    try:
        qa_chain = initialize_pipeline()
        run_cli(qa_chain)
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n❌ Environment Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        raise


if __name__ == "__main__":
    main()
