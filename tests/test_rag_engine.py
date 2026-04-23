"""Smoke tests for the retrieval pipeline.

We stub out Ollama with a deterministic word-hash embedder so the tests
run without requiring `ollama serve` or any network calls. Identical
strings always produce identical vectors, so an exact-match query has
similarity ~1.0 and an unrelated query has similarity well below the
default confidence threshold.
"""

import hashlib

import numpy as np
import pytest

import rag_engine
from rag_engine import (
    create_faiss_index,
    is_confident,
    retrieve,
    search_index,
)


DIM = 64


def _word_hash_vector(text: str) -> list[float]:
    """Deterministic bag-of-words embedding into a fixed-dim vector."""
    vec = np.zeros(DIM, dtype="float32")
    for word in text.lower().split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec.tolist()


@pytest.fixture(autouse=True)
def stub_ollama(monkeypatch):
    def fake_embeddings(model, prompt):
        return {"embedding": _word_hash_vector(prompt)}

    monkeypatch.setattr(rag_engine.ollama, "embeddings", fake_embeddings)


@pytest.fixture
def faq_index():
    questions = [
        "How do I reset my password?",
        "What is the WiFi password for the office?",
        "How do I request a new laptop?",
        "What are the cafeteria hours?",
    ]
    answers = [f"answer-{i}" for i in range(len(questions))]
    embeddings = rag_engine.embed_with_ollama(questions)
    index = create_faiss_index(embeddings)
    return questions, answers, index


def test_top_match_is_exact_question(faq_index):
    questions, _, index = faq_index
    indices, sims = retrieve(questions[0], index, top_k=3)
    assert indices[0] == 0
    assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_unrelated_query_has_low_confidence(faq_index):
    _, _, index = faq_index
    _, sims = retrieve("the quick brown fox jumps over fences", index, top_k=3)
    assert not is_confident(sims, threshold=0.6)


def test_search_index_returns_top_k(faq_index):
    questions, _, index = faq_index
    query_emb = rag_engine.embed_with_ollama([questions[2]])[0]
    sims, idxs = search_index(index, query_emb, top_k=2)
    assert len(sims) == 2 and len(idxs) == 2
    assert idxs[0] == 2


def test_confidence_threshold_gating():
    assert is_confident([0.9, 0.4], threshold=0.6)
    assert not is_confident([0.5, 0.4], threshold=0.6)
    assert not is_confident([], threshold=0.6)


def test_embed_returns_normalized_vectors():
    arr = rag_engine.embed_with_ollama(["hello world", "another phrase"])
    norms = np.linalg.norm(arr, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
