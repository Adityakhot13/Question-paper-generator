import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# =========================
# 🔑 GEMINI API
# =========================
client = genai.Client(api_key="AIzaSyCOO3Tn0YRLxasB3Lcfc2P_fukWLF4LIu0")

# =========================
# 📄 LOAD PDF
# =========================
def load_pdf(file_path):
    reader = PdfReader(file_path)
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
# 📦 FAISS INDEX
# =========================
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# =========================
# 🔍 RETRIEVAL
# =========================
def retrieve(query, chunks, index, k=2):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]

# =========================
# 🧠 GEMINI GENERATION
# =========================
def generate_questions(context):

    prompt = f"""
You are an expert university exam paper setter.

Generate a CLEAN and STRICTLY formatted question paper.

RULES:
- Only Section A, Section B, Section C
- No extra sections
- No repetition
- Clean formatting

CONTENT:
{context}

FORMAT:

Section A: MCQs (5)
- Each question must have:
  Question
  a) option
  b) option
  c) option
  d) option
  Correct Answer: ...

Section B: Short Answer (5)
- One line questions

Section C: Long Answer (3)
- Detailed questions
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# =========================
# 🚀 MAIN
# =========================
def main():

    pdf_path = r"C:\Users\adikh\OneDrive\Desktop\RAG\notes.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building index...")
    index = build_index(embeddings)

    print("Retrieving content...")
    context_chunks = retrieve(
        "Generate exam questions",
        chunks,
        index,
        k=2   # 🔥 reduces token usage
    )

    # 🔥 limit size (important)
    context = "\n\n".join(context_chunks)[:2000]

    print("Generating question paper...\n")
    result = generate_questions(context)

    print("\n===== FINAL QUESTION PAPER =====\n")
    print(result)

# =========================
if __name__ == "__main__":
    main()