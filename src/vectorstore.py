from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model once (downloads ~80MB first time)
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Create in-memory ChromaDB client
chroma_client = chromadb.Client()


def build_vectorstore(chunks):
    """
    Converts chunks to embeddings and stores them in ChromaDB.
    Returns the collection object for later searching.
    """
    # Create a fresh collection
    collection = chroma_client.get_or_create_collection(name="project13")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"], "chunk_num": chunk["chunk_num"]} for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(texts)} chunks in ChromaDB!")
    return collection


def search_vectorstore(collection, query, top_k=5):
    """
    Searches the vector store for chunks similar to the query.
    Returns top_k most relevant chunks.
    """
    print(f"\nSearching for: '{query}'")
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "score": results["distances"][0][i]
        })

    return retrieved


# Test it
if __name__ == "__main__":
    from loader import load_documents
    from chunker import chunk_documents

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    collection = build_vectorstore(chunks)

    # Test search
    results = search_vectorstore(collection, "what is machine learning?")
    print(f"\nTop results:")
    for r in results:
        print(f"\nSource: {r['source']} | Page: {r['page']} | Score: {r['score']:.4f}")
        print(f"Text: {r['text'][:200]}")