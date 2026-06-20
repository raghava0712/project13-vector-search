from sentence_transformers import CrossEncoder

# Load reranker model once (downloads ~80MB first time)
print("Loading reranker model...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker model loaded!")


def rerank(query, retrieved_chunks, top_k=3):
    """
    Takes retrieved chunks and reranks them using a cross-encoder.
    Cross-encoder scores each (query, chunk) pair together — much more accurate.
    Returns top_k best chunks after reranking.
    """
    if not retrieved_chunks:
        return []

    # Create (query, chunk_text) pairs for cross-encoder
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]

    # Score each pair
    print(f"Reranking {len(pairs)} chunks...")
    scores = reranker_model.predict(pairs)

    # Attach scores to chunks
    for i, chunk in enumerate(retrieved_chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by rerank score (higher = better)
    reranked = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)

    # Return top_k
    top_chunks = reranked[:top_k]

    print(f"Top {top_k} chunks after reranking:")
    for i, chunk in enumerate(top_chunks):
        print(f"  Rank {i+1} | Score: {chunk['rerank_score']:.4f} | Source: {chunk['source']} | Page: {chunk['page']}")

    return top_chunks


# Test it
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from vectorstore import build_vectorstore, search_vectorstore

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    collection = build_vectorstore(chunks)

    query = "what is machine learning?"

    # Step 1: Vector search - get top 5
    retrieved = search_vectorstore(collection, query, top_k=5)
    print(f"\nBefore reranking - {len(retrieved)} chunks retrieved")

    # Step 2: Rerank - pick best 3
    reranked = rerank(query, retrieved, top_k=3)
    print(f"\nAfter reranking - top {len(reranked)} chunks selected")
    print(f"\nBest chunk text:\n{reranked[0]['text'][:300]}")