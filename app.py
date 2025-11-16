import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import re

# --------------------------
# Page Design
# --------------------------
st.set_page_config(
    page_title="Searchable PDF Q&A",
    layout="wide",
    page_icon="📄"
)

st.markdown("""
<style>
    .result-box {
        background-color: #f7f7f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4A90E2;
        margin-bottom: 15px;
    }
    .context-text {
        color: #333;
        font-size: 14px;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    mark {
        background-color: yellow;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load PDFs + Build Index
# --------------------------
@st.cache_resource
def load_and_process():
    pdf_paths = [
        "data/2506.02153v2.pdf",
        "data/reasoning_models_paper.pdf"
    ]

    full_text = ""
    page_texts = []   # store text page-wise for exact match

    for path in pdf_paths:
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
                page_texts.append((page_num + 1, text))   # page number + text

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(full_text)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return chunks, embeddings, index, model, page_texts

chunks, embeddings, index, model, page_texts = load_and_process()

# --------------------------
# Exact Keyword Search Function
# --------------------------
def keyword_search(query, page_texts):
    results = []
    query_lower = query.lower()
    for page_num, text in page_texts:
        if query_lower in text.lower():
            # Highlight the keyword in the text
            highlighted_text = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
            results.append((page_num, highlighted_text))
    return results

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("📄 PDF Info")
st.sidebar.write(f"**Total Chunks:** {len(chunks)}")
st.sidebar.write(f"**Embedding Model:** MiniLM-L6-v2")
st.sidebar.write("**Vector DB:** FAISS (Local)")
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ for your ML Skill Test")

# --------------------------
# Main UI
# --------------------------
st.title("🔎 Searchable PDF Q&A App")
st.subheader("Ask anything about the uploaded research papers.")

query = st.text_input("Enter your question:", placeholder="e.g., What is the main idea of the paper?")

if query:
    start_time = time.time()

    # --- Exact keyword search ---
    exact_results = keyword_search(query, page_texts)
    
    if exact_results:
        elapsed = round((time.time() - start_time) * 1000, 2)
        st.markdown(f"### ⏱ Response Time: `{elapsed} ms`")
        st.markdown("### 🔍 Exact Match Results:")

        for page_num, highlighted_text in exact_results:
            st.markdown(f"#### Page {page_num}")
            st.markdown(
                f"<div class='result-box'><div class='context-text'>{highlighted_text}</div></div>",
                unsafe_allow_html=True
            )
    else:
        # --- Fall back to FAISS semantic search ---
        query_embedding = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, 3)

        elapsed = round((time.time() - start_time) * 1000, 2)
        st.markdown(f"### ⏱ Response Time: `{elapsed} ms`")
        st.markdown("### 🔍 Top Relevant Chunks:")

        for rank, idx in enumerate(I[0]):
            similarity = round(1 / (1 + D[0][rank]), 3)
            st.markdown(f"#### Result {rank+1} — Similarity: **{similarity}**")
            st.markdown(
                f"<div class='result-box'><div class='context-text'>{chunks[idx]}</div></div>",
                unsafe_allow_html=True
            )
