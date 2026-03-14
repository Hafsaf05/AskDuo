"""
loader.py
---------
Handles loading documents from the knowledge base (text files).
Supports plain .txt files and returns LangChain Document objects.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader


def load_documents(file_path: str) -> List[Document]:
    """
    Load documents from a text file and return a list of LangChain Documents.

    Args:
        file_path (str): Path to the source document file.

    Returns:
        List[Document]: A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file is empty or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[Loader] Document not found at path: '{file_path}'\n"
            f"Please ensure 'data/documents.txt' exists before running."
        )

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    if not documents:
        raise ValueError(
            f"[Loader] No content found in file: '{file_path}'"
        )

    print(f"[Loader] ✓ Loaded {len(documents)} document(s) from '{file_path}'")
    return documents