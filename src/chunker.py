def chunk_documents(documents, chunk_size=500, overlap=50):
    """
    Splits large document pages into smaller overlapping chunks.
    
    chunk_size = max characters per chunk
    overlap = how many characters repeat between chunks (for context continuity)
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]

        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_num": chunk_num
                })
                chunk_num += 1

            start = end - overlap  # overlap for context continuity

    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# Test it
if __name__ == "__main__":
    from loader import load_documents

    docs = load_documents("data")
    chunks = chunk_documents(docs)

    print(f"\nFirst chunk preview:")
    print(f"Source: {chunks[0]['source']} | Page: {chunks[0]['page']} | Chunk: {chunks[0]['chunk_num']}")
    print(f"Text: {chunks[0]['text'][:300]}")
    print(f"\nTotal characters in first chunk: {len(chunks[0]['text'])}")