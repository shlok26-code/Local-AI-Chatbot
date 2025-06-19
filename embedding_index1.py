import chromadb
from chromadb.utils import embedding_functions

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    # ChromaDB handles embedding if you provide an embedding function, so just return chunks here
    return chunks

def build_chromadb_collection(chunks, collection_name="my_collection", model_name="all-MiniLM-L6-v2", persist_directory=None):
    # Set up the embedding function using SentenceTransformer
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    # Set up the ChromaDB client (persistent if directory provided)
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client()

    # Create or get a collection with the embedding function
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    # Prepare document ids
    ids = [f"doc_{i}" for i in range(len(chunks))]
    # Add chunks to the collection
    collection.add(ids=ids, documents=chunks)

    return collection