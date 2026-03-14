"""
embeddings.py
-------------
Configures and returns the OpenAI embedding model used to convert
document chunks and queries into dense vector representations.

Model: text-embedding-3-small
  - 1536-dimensional output vectors
  - Fast, cost-effective, and high-quality for RAG use cases
  - Supports up to 8,191 tokens per input
"""

import os
from langchain_openai import OpenAIEmbeddings


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initialize and return the OpenAI embeddings model.

    Reads the OPENAI_API_KEY from the environment (loaded via dotenv).
    Uses the 'text-embedding-3-small' model for efficient, high-quality
    semantic embeddings.

    Returns:
        OpenAIEmbeddings: A configured LangChain embeddings instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "[Embeddings] OPENAI_API_KEY is not set.\n"
            "Please add it to your .env file:\n"
            "  OPENAI_API_KEY=your_key_here"
        )

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    print("[Embeddings] ✓ Initialized OpenAI embeddings (text-embedding-3-small)")
    return embedding_model
