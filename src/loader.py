import fitz  # PyMuPDF
import os

def load_documents(folder_path):
    """
    Reads all PDF and TXT files from the data/ folder.
    Returns a list of dictionaries with text and source info.
    """
    documents = []

    files = os.listdir(folder_path)
    print(f"Files found: {files}")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        print(f"Processing: {filepath}")

        # Load PDF files
        if filename.lower().endswith(".pdf"):
            try:
                doc = fitz.open(filepath)
                print(f"  Pages in PDF: {len(doc)}")
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": filename,
                            "page": page_num + 1
                        })
                doc.close()
            except Exception as e:
                print(f"  Error: {e}")

        # Load TXT files
        elif filename.lower().endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": filename,
                            "page": 1
                        })
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\nTotal loaded: {len(documents)} pages")
    return documents


# Test it
if __name__ == "__main__":
    docs = load_documents("data")
    for doc in docs[:2]:
        print(f"\nSource: {doc['source']} | Page: {doc['page']}")
        print(f"Text preview: {doc['text'][:200]}")