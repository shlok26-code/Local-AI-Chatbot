import numpy as np
from pdf_extraction import extract_text_from_pdf
from text_chunking import chunk_text
from embedding_index import embed_chunks, build_faiss_index
from semantic_search import search_index
from semantic_search import answer_question
from sentence_transformers import SentenceTransformer

# Step 1: Extract and chunk
pdf_path = r"C:\Users\mynam\Downloads\Um_dearborn\Q&A Bot\Project1\QABot1\Text\AI.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# Step 2: Embed and index
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_chunks(chunks, model_name="all-MiniLM-L6-v2")
index = build_faiss_index(np.array(embeddings))

# Step 3: User asks a question
question = input("Ask a question: ")
relevant_chunks = search_index(question, model, index, chunks, top_k=3)
context = " ".join(relevant_chunks)

# Step 4: Get answer
answer = answer_question(question, context)
print(f"Answer: {answer}")