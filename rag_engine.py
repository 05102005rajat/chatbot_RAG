import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
import ollama

from config import (
    EMBEDDING_MODEL,
    RAG_MODEL,
    EMBEDDING_CACHE_FOLDER,
    CONFIDENCE_THRESHOLD,
    SYSTEM_PROMPT,
    LOW_CONFIDENCE_FALLBACK,
)


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama daemon can't be reached or a model is missing."""


def _embed_one(text: str) -> list[float]:
    try:
        return ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)["embedding"]
    except Exception as e:
        raise OllamaUnavailable(
            f"Could not get embedding from Ollama (model={EMBEDDING_MODEL}). "
            f"Is `ollama serve` running and is the model pulled? Original error: {e}"
        ) from e


def embed_with_ollama(texts: list[str], max_workers: int = 8) -> np.ndarray:
    """Embed a list of texts in parallel and return a normalized float32 matrix."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        vectors = list(pool.map(_embed_one, texts))
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, index_path: str) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)


def _cache_path_for(source_path: str) -> str:
    """Return a stable cache path for a given source FAQ file."""
    h = hashlib.sha256()
    with open(source_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()[:16]
    base = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(EMBEDDING_CACHE_FOLDER, f"{base}.{digest}.npy")


def get_or_build_embeddings(texts: list[str], source_path: str) -> np.ndarray:
    """Return cached embeddings for a source file, or compute and cache them."""
    cache_path = _cache_path_for(source_path)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached
    embeddings = embed_with_ollama(texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def search_index(
    index: faiss.Index, query_embedding: np.ndarray, top_k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    similarities, indices = index.search(query, top_k)
    return similarities[0], indices[0]


def _build_prompt(
    question: str,
    questions: list[str],
    answers: list[str],
    relevant_idxs: np.ndarray,
    department: str,
) -> list[dict]:
    context = "\n\n".join(
        f"Q: {questions[i]}\nA: {answers[i]}"
        for i in relevant_idxs
        if 0 <= i < len(questions)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(department=department)},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]


def retrieve(
    question: str,
    index: faiss.Index,
    top_k: int = 3,
) -> tuple[list[int], list[float]]:
    """Embed the question and return (faq_indices, similarities) for the top_k matches."""
    query_emb = embed_with_ollama([question])[0]
    similarities, indices = search_index(index, query_emb, top_k)
    return indices.tolist(), similarities.tolist()


def is_confident(similarities: list[float], threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    return bool(similarities) and similarities[0] >= threshold


def stream_answer(
    question: str,
    questions: list[str],
    answers: list[str],
    relevant_idxs: list[int],
    department: str,
):
    """Yield response chunks (strings) from the LLM."""
    messages = _build_prompt(question, questions, answers, relevant_idxs, department)
    try:
        for chunk in ollama.chat(model=RAG_MODEL, messages=messages, stream=True):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
    except Exception as e:
        raise OllamaUnavailable(
            f"Could not stream chat completion from Ollama (model={RAG_MODEL}). "
            f"Is `ollama serve` running? Original error: {e}"
        ) from e


def fallback_message(department: str) -> str:
    return LOW_CONFIDENCE_FALLBACK.format(department=department)
