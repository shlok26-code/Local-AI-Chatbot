from pdf_extraction import extract_text_from_pdf
from text_chunking import chunk_text
from embedding_index1 import embed_chunks, build_chromadb_collection
from semantic_search1 import search_index, answer_question
from sentence_transformers import SentenceTransformer

# Step 1: Extract and chunk
pdf_path = r"C:\Users\mynam\Downloads\Um_dearborn\Q&A Bot\Project1\QABot1\Text\AI.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# Step 2: Build ChromaDB collection
collection = build_chromadb_collection(
    chunks,
    collection_name="my_collection",
    model_name="all-MiniLM-L6-v2"
)

# Step 3: User asks a question
question = input("Ask a question: ")

# Step 4: Search relevant chunks using ChromaDB
relevant_chunks = search_index(question, collection, top_k=3)
context = " ".join(relevant_chunks)

# Step 5: Get answer
answer = answer_question(question, context)
print(f"Answer: {answer}")