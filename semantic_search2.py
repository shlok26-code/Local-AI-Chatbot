from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load model and tokenizer once at the top
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def search_index(question, collection, top_k=3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results['documents'][0]

def answer_question(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=False
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
   # return answer.split("Answer:")[-1].strip()
    answer_text = answer.split("Answer:")[-1].strip()
    
    # Trim to the last full sentence
    sentences = re.split(r'([.!?])', answer_text)
    result = ""
    for i in range(0, len(sentences)-1, 2):
        result += sentences[i] + sentences[i+1]
    return result.strip() if result else answer_text