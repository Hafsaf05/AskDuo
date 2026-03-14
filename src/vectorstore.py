"""
vectorstore.py
--------------
Builds and persists a Chroma vector database from document chunks,
or loads an existing one. Returns a retriever for similarity search.

Chroma stores vectors on disk under the 'vectorstore/' directory,
so re-runs skip re-embedding unless the store is cleared.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Path where Chroma persists its index on disk
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
COLLECTION_NAME = "askduo_knowledge_base"

# Number of top-k chunks to retrieve per query
DEFAULT_K = 4


def build_vectorstore(
    chunks: List[Document],
    embedding_model: OpenAIEmbeddings,
    persist_directory: str = VECTORSTORE_DIR,
) -> Chroma:
    """
    Create a Chroma vector store from document chunks and persist it to disk.

    Each chunk is converted into an embedding vector via the provided model,
    then stored in Chroma's local index. Subsequent runs load from disk.

    Args:
        chunks (List[Document]): Chunked documents to embed and store.
        embedding_model (OpenAIEmbeddings): The embedding model to use.
        persist_directory (str): Directory where Chroma persists its index.

    Returns:
        Chroma: A populated and persisted Chroma vector store.
    """
    os.makedirs(persist_directory, exist_ok=True)

    print(f"[VectorStore] ⚙  Embedding {len(chunks)} chunks into Chroma...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    print(
        f"[VectorStore] ✓ Vector store built and persisted at "
        f"'{persist_directory}'"
    )
    return vectorstore


def load_vectorstore(
    embedding_model: OpenAIEmbeddings,
    persist_directory: str = VECTORSTORE_DIR,
) -> Chroma:
    """
    Load an existing Chroma vector store from disk.

    Args:
        embedding_model (OpenAIEmbeddings): Must match the model used to build
                                             the store originally.
        persist_directory (str): Directory where Chroma was persisted.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    print(f"[VectorStore] ✓ Loaded existing vector store from '{persist_directory}'")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def get_retriever(
    vectorstore: Chroma,
    k: int = DEFAULT_K,
):
    """
    Return a similarity-based retriever from the vector store.

    Args:
        vectorstore (Chroma): A populated Chroma instance.
        k (int): Number of top similar chunks to retrieve per query.

    Returns:
        VectorStoreRetriever: A LangChain retriever instance.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    print(f"[VectorStore] ✓ Retriever ready (top-{k} similarity search)")
    return retriever