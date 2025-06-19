import numpy as np
from transformers import pipeline

def search_index(question, model, index, chunks, top_k=3):
    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in I[0]]



def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)  # 0 for GPU, -1 for CPU
    result = qa_pipeline(question=question, context=context)
    return result['answer']