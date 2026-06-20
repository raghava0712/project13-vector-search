from sentence_transformers import CrossEncoder

# NLI model checks if answer is supported by context
print("Loading hallucination detector...")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
print("Hallucination detector loaded!")

# Confidence threshold - below this we say "I don't know"
CONFIDENCE_THRESHOLD = 0.3


def check_confidence(reranked_chunks):
    """
    Checks if retrieved chunks are confident enough to answer.
    If best rerank score is too low, we shouldn't answer.
    """
    if not reranked_chunks:
        return False, 0.0

    best_score = reranked_chunks[0]["rerank_score"]

    # Normalize score to 0-1 range roughly
    # rerank scores above 0 mean relevant, below 0 mean not relevant
    is_confident = best_score > CONFIDENCE_THRESHOLD

    print(f"Confidence check: best score = {best_score:.4f} | confident = {is_confident}")
    return is_confident, best_score


def check_hallucination(answer, reranked_chunks):
    """
    Uses NLI model to check if the answer is supported by the context.
    Returns: supported, contradicted, or neutral
    """
    if not reranked_chunks:
        return "neutral", 0.0

    # Combine top chunks as context
    context = " ".join([chunk["text"] for chunk in reranked_chunks[:2]])

    # NLI checks: does context ENTAIL (support) the answer?
    scores = nli_model.predict([(context, answer)])

    # Labels: 0=contradiction, 1=neutral, 2=entailment
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    predicted_label = scores[0].argmax()
    label = label_map[predicted_label]
    confidence = float(scores[0].max())

    print(f"Hallucination check: {label} (confidence: {confidence:.4f})")
    return label, confidence


def safe_generate(query, reranked_chunks, answer):
    """
    Wraps answer generation with confidence + hallucination checks.
    Returns final safe answer with flags.
    """
    # Step 1: Check if we're confident enough to answer
    is_confident, score = check_confidence(reranked_chunks)

    if not is_confident:
        return {
            "answer": "I don't have enough relevant information to answer this question confidently.",
            "sources": [],
            "hallucination_check": "skipped - low confidence",
            "confidence_score": score,
            "is_safe": False
        }

    # Step 2: Check if answer is supported by context
    label, hal_confidence = check_hallucination(answer, reranked_chunks)

    sources = list(set([f"{c['source']} (Page {c['page']})" for c in reranked_chunks]))

    if label == "contradiction":
        return {
            "answer": "I found conflicting information. Please verify with the source documents directly.",
            "sources": sources,
            "hallucination_check": "contradiction detected",
            "confidence_score": score,
            "is_safe": False
        }

    return {
        "answer": answer,
        "sources": sources,
        "hallucination_check": label,
        "confidence_score": score,
        "is_safe": True
    }


# Test it
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from vectorstore import build_vectorstore, search_vectorstore
    from reranker import rerank
    from llm import generate_answer

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    collection = build_vectorstore(chunks)

    # Test 1 - question with good answer
    print("\n" + "="*50)
    print("TEST 1: Good question")
    print("="*50)
    query = "what is machine learning?"
    retrieved = search_vectorstore(collection, query, top_k=5)
    reranked = rerank(query, retrieved, top_k=3)
    answer, _ = generate_answer(query, reranked)
    result = safe_generate(query, reranked, answer)
    print(f"ANSWER: {result['answer']}")
    print(f"SAFE: {result['is_safe']}")
    print(f"HALLUCINATION CHECK: {result['hallucination_check']}")
    print(f"SOURCES: {result['sources']}")

    # Test 2 - question with no relevant info
    print("\n" + "="*50)
    print("TEST 2: Question outside documents")
    print("="*50)
    query2 = "what is the capital of france?"
    retrieved2 = search_vectorstore(collection, query2, top_k=5)
    reranked2 = rerank(query2, retrieved2, top_k=3)
    answer2, _ = generate_answer(query2, reranked2)
    result2 = safe_generate(query2, reranked2, answer2)
    print(f"ANSWER: {result2['answer']}")
    print(f"SAFE: {result2['is_safe']}")
    print(f"HALLUCINATION CHECK: {result2['hallucination_check']}")