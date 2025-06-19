from gpt4all import GPT4All
import re


# Load GPT4All model 
MODEL_PATH=r"C:\Users\mynam\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
gpt4all_model = GPT4All(model_name=MODEL_PATH)

def search_index(question, collection, top_k=3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results['documents'][0]

def answer_question(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    with gpt4all_model.chat_session():
        answer = gpt4all_model.generate(prompt, max_tokens=128, temp=0.2)
    # gpt4all doesn't prepend 'Answer:' so just return the output
    answer_text = answer.strip()
   # return answer.strip()

    # Trim to the last full sentence
    sentences = re.split(r'([.!?])', answer_text)
    result = ""
    for i in range(0, len(sentences)-1, 2):
        result += sentences[i] + sentences[i+1]
    return result.strip() if result else answer_text