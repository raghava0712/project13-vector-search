import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import build_pipeline, run_pipeline

st.set_page_config(
    page_title="Project 13 - Vector Search Pipeline",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Vector Search & Reranking Pipeline")
st.caption("Project 13 | Semantic Search + Reranking + Hallucination Detection")

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

    # Reload button after upload
    if st.button("🔄 Reload Pipeline with new docs"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**Pipeline steps:**")
    st.markdown("1. 📄 Load & chunk documents")
    st.markdown("2. 🔢 Create embeddings")
    st.markdown("3. 🔍 Vector search (top 5)")
    st.markdown("4. 📊 Rerank (top 3)")
    st.markdown("5. 🤖 LLaMA generates answer")
    st.markdown("6. 🛡️ Hallucination check")

# ── Build pipeline ────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline... please wait")
def get_pipeline():
    return build_pipeline("data")

collection = get_pipeline()

# ── Main chat area ────────────────────────────────────
st.subheader("💬 Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask anything about your documents..."):

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            result = run_pipeline(query, collection)

        st.markdown(result["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            safe_icon = "✅" if result["is_safe"] else "⚠️"
            st.metric("Safety", f"{safe_icon} {'Safe' if result['is_safe'] else 'Unsafe'}")
        with col2:
            st.metric("Confidence", f"{result['confidence_score']:.2f}")
        with col3:
            st.metric("Hallucination", result["hallucination_check"])

        if result["sources"]:
            with st.expander("📚 Sources"):
                for source in result["sources"]:
                    st.write(f"• {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })