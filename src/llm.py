import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env file

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query, reranked_chunks):
    """
    Takes the query and top reranked chunks, sends to LLaMA via Groq,
    and returns an answer grounded in the retrieved context.
    """
    if not reranked_chunks:
        return "I don't have enough information to answer this question.", []

    # Build context from top chunks
    context = ""
    sources = []
    for i, chunk in enumerate(reranked_chunks):
        context += f"\n[Source {i+1}: {chunk['source']}, Page {chunk['page']}]\n"
        context += chunk["text"] + "\n"
        sources.append(f"{chunk['source']} (Page {chunk['page']})")

    # Build prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided documents."
Always be concise and accurate.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    print("Sending to LLaMA via Groq...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512
    )

    answer = response.choices[0].message.content.strip()
    return answer, list(set(sources))


# Test it
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents
    from vectorstore import build_vectorstore, search_vectorstore
    from reranker import rerank

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    collection = build_vectorstore(chunks)

    query = "what is machine learning?"

    # Full pipeline
    retrieved = search_vectorstore(collection, query, top_k=5)
    reranked = rerank(query, retrieved, top_k=3)
    answer, sources = generate_answer(query, reranked)

    print(f"\n{'='*50}")
    print(f"QUESTION: {query}")
    print(f"{'='*50}")
    print(f"ANSWER: {answer}")
    print(f"\nSOURCES: {sources}")