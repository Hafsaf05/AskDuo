"""
chunker.py
----------
Splits loaded documents into smaller, overlapping chunks using
RecursiveCharacterTextSplitter for optimal retrieval performance.

Chunk size and overlap are tuned to balance context richness with
embedding quality.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunking configuration
DEFAULT_CHUNK_SIZE = 500          # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 100       # Overlap between consecutive chunks
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on paragraph
    breaks first, then newlines, then sentences, then words — preserving
    as much semantic coherence as possible per chunk.

    Args:
        documents (List[Document]): Raw documents from the loader.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of smaller document chunks, each retaining
                        the original metadata (e.g., source file path).

    Raises:
        ValueError: If no documents are provided.
    """
    if not documents:
        raise ValueError("[Chunker] No documents provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("[Chunker] Document splitting produced no chunks.")

    print(
        f"[Chunker] ✓ Split {len(documents)} document(s) into "
        f"{len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks