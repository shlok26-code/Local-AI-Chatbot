# Local-AI-Chatbot

An Interactive local chatbot using LLMs to provide answers based on local documents.

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot that runs locally using a Streamlit interface and allows users to interact with their own documents using state-of-the-art Large Language Models (LLMs). The chatbot intelligently retrieves relevant document chunks and uses LLMs to generate accurate and context-aware responses.

## 🔍 What is RAG?

**RAG** is a powerful NLP approach that combines:

* **Retrieval**: Find relevant content from the knowledge base (your documents) using vector similarity.
* **Augmented Generation**: Feed the retrieved context into a generative LLM to produce meaningful and context-aware answers.

##  Features

* Supports **two LLMs**:

  * `Mistral-7B-Instruct-v0.2` via Hugging Face
  * `Meta-Llama-3-8B-Instruct` via GPT4All 
* Dynamic model selection in the UI
* Uses **ChromaDB** for vector storage (with optional FAISS backend)
* Built using **Streamlit** for an intuitive and interactive frontend
* Prior experimentation with extractive model `deepset/roberta-base-squad2`

## 🗂️ Repository Structure

| File                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `app_final.py`        | Main Streamlit application for interactive chatbot            |
| `main_final.py`       | Core logic for document retrieval and LLM response generation |
| `embedding_index.py`  | Indexing and retrieval using FAISS embeddings                 |
| `embedding_index1.py` | Indexing and retrieval using ChromaDB embeddings              |
| `main_faiss.py`       | RAG with FAISS and `roberta-base-squad2`                      |
| `main_chromaDB.py`    | RAG with ChromaDB and `roberta-base-squad2`                   |
| `requirements.txt`    | List of dependencies                                          |

## 📂 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/shlok26-code/Local-AI-Chatbot.git
cd Local-AI-Chatbot
```

2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. **Prepare your document**:

   * Create a directory called `Text/`
   * Place your local PDF document inside it
   * Update the path to the PDF document in `main_final.py` accordingly

4. Run the chatbot:

```bash
streamlit run app_final.py
```

## 🧪 Optional Experiments

To test with extractive QA using `deepset/roberta-base-squad2`, run:

### FAISS Version

```bash
python main_faiss.py
```

### ChromaDB Version

```bash
python main_chromaDB.py
```

## 📦 Requirements

All required packages are listed in `requirements.txt`. Key libraries include:

* `streamlit`
* `transformers`
* `sentence-transformers`
* `chromadb`
* `faiss-cpu` *(for FAISS backend)*
* `gpt4all`
* `huggingface_hub`

## ✍️ Author

**Shlok Sharma**

## ⭐ Future Improvements

* Add more model options
* Enhance document management and upload support
* Implement summarization capabilities
* Enable multi-file/document support

## 📌 Note

If you wish to try this project, ensure you create a `Text/` directory and place a local PDF file inside it. Then, **update the path in `main_final.py`** to reflect the correct file name and location.

---

Give the project a ⭐ if you found it useful!
