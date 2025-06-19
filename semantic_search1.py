from transformers import pipeline

def search_index(question, collection, top_k=3):
    """
    Searches the ChromaDB collection for the most relevant chunks to the question.
    """
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    # results['documents'] is a list of lists (since you can query multiple texts at once)
    # We want the list of documents for the first (and only) query
    return results['documents'][0]

def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)  # 0 for GPU, -1 for CPU
    result = qa_pipeline(question=question, context=context)
    return result['answer']