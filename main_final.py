from pdf_extraction import extract_text_from_pdf
from text_chunking import chunk_text
from embedding_index1 import build_chromadb_collection
from semantic_search2 import search_index, answer_question
from semantic_search3 import answer_question as answer_question1

# Step 1: Extract and chunk only once
pdf_path = r"C:\Users\mynam\Downloads\Um_dearborn\Q&A Bot\Project1\QABot1\Text\AI.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# Step 2: Build ChromaDB collection only once
collection = build_chromadb_collection(
    chunks,
    collection_name="my_collection",
    model_name="all-MiniLM-L6-v2"
)

def get_answer(question, model_choice):
    # Step 3: Search relevant chunks using ChromaDB
    relevant_chunks = search_index(question, collection, top_k=3)
    context = " ".join(relevant_chunks)

    # Step 4: Get answer, select function based on model_choice
    if model_choice == "1":
        answer = answer_question(question, context)
    elif model_choice == "2":
        answer = answer_question1(question, context)
    else:
        answer = "Model selection error. Please restart and select a valid model."
    return answer

if __name__ == "__main__":
    # For testing in command line
    print("Select model:")
    print("1. Mistral-7B-Instruct-v0.2")
    print("2. Meta-Llama-3-8B-Instruct")
    model_choice = input("Enter 1 or 2: ")
    if model_choice not in ["1", "2"]:
        print("Invalid model selection.")
    else:
        question = input("Ask a question: ")
        answer = get_answer(question, model_choice)
        print(f"Answer: {answer}")