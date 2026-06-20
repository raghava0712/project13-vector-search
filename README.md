# 🔍 Vector Search & Reranking Pipeline with Hallucination Detection
> Project 13 | NVIDIA DLI Advanced Deep Learning Projects

A production-ready semantic search pipeline that combines vector search, 
cross-encoder reranking, and NLI-based hallucination detection to deliver 
accurate, trustworthy answers from your documents.

🚀 **Live Demo:** [Click here to try it](https://project13-vector-search-sddcjwcqvcc3psvnshteya.streamlit.app/)

---

## 🧠 What it does

Upload any PDF or TXT document and ask questions in natural language. 
The system retrieves relevant chunks, reranks them for accuracy, generates 
an answer using LLaMA 3.1, and checks for hallucinations before showing 
the response.

---

## ⚙️ Pipeline Architecture
---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| LangChain | Pipeline orchestration |
| ChromaDB | Vector database (in-memory) |
| sentence-transformers | Text embeddings |
| cross-encoder/ms-marco | Reranking model |
| cross-encoder/nli-deberta-v3 | Hallucination detection |
| Groq API + LLaMA 3.1 | Answer generation (free) |
| Streamlit | Web UI + deployment |
| PyMuPDF | PDF reading |

---

## 📁 Project Structure
---

## 📊 Evaluation Results

Tested on 10 questions across multiple documents:

| Metric | Score |
|---|---|
| Safe answers | 5/10 |
| Entailment (supported) | 5/10 |
| Avg keyword score | 0.37 |
| Hallucination rate | 50% |
| Outside-domain refusal | ✅ Correct |

> Note: Low scores on some questions are due to limited document content,
> not pipeline failures. The system correctly refuses to answer when 
> confidence is low rather than hallucinating.

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/raghava0712/project13-vector-search.git
cd project13-vector-search

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# Run the app
streamlit run app/app.py
```

---

## 🔑 Get Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Paste it in your `.env` file

---

## 👨‍💻 Author

**Raghava** | Data Science Student  
Built as part of NVIDIA DLI Advanced Deep Learning Projects
