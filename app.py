import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os
import time

# =========================
# 🔑 API (SAFE)
# =========================
# API_KEY = os.getenv("API_KEY")

# if not API_KEY:
#     st.error("API key not found! Add it in Streamlit Secrets.")
#     st.stop()

# client = genai.Client(api_key=API_KEY)
client = genai.Client(api_key="AIzaSyCOO3Tn0YRLxasB3Lcfc2P_fukWLF4LIu0")

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
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

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
# 🧠 GENERATE (FIXED)
# =========================
def generate_questions(context):

    prompt = f"""
You are an expert university exam paper setter.

STRICT RULES:
- Only Section A, Section B, Section C
- Do NOT repeat sections
- Do NOT add extra sections
- Keep format clean

CONTENT:
{context}

FORMAT:

Section A: MCQs (5)
Each question must include:
- Question
- a) option
- b) option
- c) option
- d) option
- Correct Answer

Section B: Short Answer (5)

Section C: Long Answer (3)
"""

    for i in range(3):  # retry logic
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except Exception as e:
            time.sleep(5)

    return "⚠️ Error: Unable to generate question paper. Try again."

# =========================
# 🎨 UI
# =========================
st.title("📘 AI Question Paper Generator")

uploaded_file = st.file_uploader("Upload Notes (PDF)", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    if st.button("Generate Question Paper"):

        with st.spinner("Processing..."):

            try:
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

                # 🔥 control tokens
                context = "\n\n".join(context_chunks)[:1500]

                result = generate_questions(context)

                st.subheader("Generated Question Paper")
                st.text_area("Output", result, height=400)

                st.download_button(
                    label="Download Question Paper",
                    data=result,
                    file_name="question_paper.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")
