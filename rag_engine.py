import numpy as np
import faiss
import ollama
import os
from config import EMBEDDING_MODEL, RAG_MODEL

# 🔹 Step 1: Embed text using Ollama
def embed_with_ollama(texts):
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embeddings.append(response['embedding'])
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)  # normalize for cosine similarity
    return embeddings

# 🔹 Step 2: Create FAISS index using cosine similarity
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity
    index.add(embeddings)
    return index

# 🔹 Step 3: Save and load FAISS index
def save_faiss_index(index, index_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# 🔹 Step 4: Search index and return similarities + indices
def search_index(index, query_embedding, top_k=3):
    faiss.normalize_L2(query_embedding.reshape(1, -1))  # normalize query
    similarities, indices = index.search(np.array([query_embedding]), top_k)
    return similarities[0], indices[0]


# 🔹 Step 6: Generate RAG answer with confidence check
def generate_rag_answer(question, index, questions, answers, top_k=3, confidence_threshold=0.6):
    query_emb = embed_with_ollama([question])[0]
    similarities, relevant_idxs = search_index(index, query_emb, top_k)

    context = "\n\n".join([f"Q: {questions[i]}\nA: {answers[i]}" for i in relevant_idxs])
    prompt = f"Context:\n{context}\n\nBased on the above, answer this question:\n{question}, if the question is out of context, respond with 'I don't know'."
    response = ollama.chat(model=RAG_MODEL, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
