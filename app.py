import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# =========================
# 🔑 API
# =========================
client = genai.Client(api_key="AIzaSyCOO3Tn0YRLxasB3Lcfc2P_fukWLF4LIu0")
# import os
# client = genai.Client(api_key=os.getenv("AIzaSyCOO3Tn0YRLxasB3Lcfc2P_fukWLF4LIu0"))

# =========================
# 📄 LOAD PDF
# =========================
def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# =========================
# ✂️ CHUNKING
# =========================
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# =========================
# 🔢 EMBEDDINGS
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    return embed_model.encode(chunks)

# =========================
# 📦 INDEX
# =========================
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# =========================
# 🔍 RETRIEVE
# =========================
def retrieve(query, chunks, index, k=2):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]

# =========================
# 🧠 GENERATE
# =========================
def generate_questions(context):

    prompt = f"""
You are an expert university exam paper setter.

Generate a CLEAN and STRICTLY formatted question paper.

RULES:
- Only Section A, Section B, Section C
- No repetition
- No extra sections

CONTENT:
{context}

FORMAT:

Section A: MCQs (5)
- 4 options each
- Mark correct answer

Section B: Short Answer (5)

Section C: Long Answer (3)
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",   # stable
        contents=prompt
    )

    return response.text

# =========================
# 🎨 STREAMLIT UI
# =========================
st.title("📘 AI Question Paper Generator")

uploaded_file = st.file_uploader("Upload Notes (PDF)", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    if st.button("Generate Question Paper"):

        with st.spinner("Processing..."):

            text = load_pdf(uploaded_file)
            chunks = chunk_text(text)
            embeddings = create_embeddings(chunks)
            index = build_index(embeddings)

            context_chunks = retrieve(
                "Generate exam questions",
                chunks,
                index,
                k=2
            )

            context = "\n\n".join(context_chunks)[:2000]

            result = generate_questions(context)

            st.subheader("Generated Question Paper")
            st.text_area("Output", result, height=400)

            # Download button
            st.download_button(
                label="Download Question Paper",
                data=result,
                file_name="question_paper.txt",
                mime="text/plain"
            )
