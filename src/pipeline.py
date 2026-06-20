from loader import load_documents
from chunker import chunk_documents
from vectorstore import build_vectorstore, search_vectorstore
from reranker import rerank
from llm import generate_answer
from hallucination import safe_generate


def build_pipeline(data_folder="data"):
    """
    Loads documents, chunks them, and builds vector store.
    Call this once at startup.
    """
    docs = load_documents(data_folder)
    chunks = chunk_documents(docs)
    collection = build_vectorstore(chunks)
    return collection


def run_pipeline(query, collection):
    """
    Runs the full pipeline for a single query.
    Returns a clean result dictionary.
    """
    # Step 1: Vector search
    retrieved = search_vectorstore(collection, query, top_k=5)

    # Step 2: Rerank
    reranked = rerank(query, retrieved, top_k=3)

    # Step 3: Generate answer
    answer, _ = generate_answer(query, reranked)

    # Step 4: Hallucination + confidence check
    result = safe_generate(query, reranked, answer)

    return result