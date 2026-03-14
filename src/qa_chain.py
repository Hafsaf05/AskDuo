"""
qa_chain.py
-----------
Builds the LangChain RetrievalQA chain that powers AskDuo's question-answering.

The chain:
  1. Takes a user question
  2. Retrieves the top-k relevant document chunks from Chroma
  3. Injects those chunks as context into a prompt
  4. Sends the enriched prompt to GPT-4o-mini
  5. Returns the generated answer + source documents

Model: gpt-4o-mini
  - Fast and cost-efficient
  - 128k context window
  - High quality for RAG summarization and QA tasks
"""

import os
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ── Prompt Template ────────────────────────────────────────────────────────────
# Instructs the LLM to answer only from the provided context, and to
# gracefully handle cases where the answer isn't found.

RAG_PROMPT_TEMPLATE = """You are AskDuo, a precise and helpful AI assistant.
Use ONLY the context provided below to answer the question.
If the answer is not found in the context, say:
"I don't have enough information to answer that based on the current knowledge base."

Do not make up information. Be concise, clear, and factual.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_qa_chain(retriever) -> RetrievalQA:
    """
    Construct and return a LangChain RetrievalQA chain.

    Combines:
    - ChatOpenAI (gpt-4o-mini) as the language model
    - The provided retriever for document lookup
    - A custom RAG prompt that enforces grounded, contextual answers

    Args:
        retriever: A LangChain VectorStoreRetriever from vectorstore.py.

    Returns:
        RetrievalQA: The assembled QA chain, ready to invoke.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "[QAChain] OPENAI_API_KEY is not set.\n"
            "Please add it to your .env file."
        )

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,          # Low temperature for factual, consistent answers
        openai_api_key=api_key,
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # 'stuff' = inject all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,   # Include sources in the response
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    print("[QAChain] ✓ RetrievalQA chain built (gpt-4o-mini, chain_type=stuff)")
    return qa_chain


def ask(qa_chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the QA chain and return the result.

    Args:
        qa_chain (RetrievalQA): The assembled QA chain.
        question (str): The user's question.

    Returns:
        dict: Contains 'answer' (str) and 'sources' (List[Document]).
    """
    if not question.strip():
        return {"answer": "Please enter a valid question.", "sources": []}

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result.get("result", "No answer generated."),
        "sources": result.get("source_documents", []),
    }