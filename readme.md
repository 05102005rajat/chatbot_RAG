# RAG-powered FAQ Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, FAISS, and Ollama. Users pick a department, ask a question in natural language, and get a grounded answer drawn from that department's FAQ spreadsheet.

## Features

- Semantic search over per-department FAQs with FAISS (cosine similarity)
- Local LLM responses via Ollama — nothing leaves your machine
- **Confidence threshold**: low-similarity questions get a polite "I don't know" instead of a hallucinated answer
- **Source display**: every response shows which FAQs it was grounded in, with similarity scores
- **Streaming responses** for snappier UX
- Disk-cached embeddings keyed by a hash of the source spreadsheet — change the FAQ file and the cache invalidates automatically
- Per-department FAISS index, persisted to `vector_stores/`
- Chat history saving (JSON) and loading; legacy `.txt` logs still readable

## Setup

### 1. Install Ollama and pull the models

Install Ollama from <https://ollama.com>, then:

```bash
ollama pull embeddinggemma:latest
ollama pull gemma3:1b
ollama serve
```

(Models are configurable in `config.py` — `EMBEDDING_MODEL` and `RAG_MODEL`.)

### 2. Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>).

## Adding a department

1. Drop an Excel file with `question` and `answer` columns into `faqs/`.
2. Add an entry to `DEPARTMENT_FILES` in `config.py`.
3. Restart the app — the index is built lazily on first selection.

## Tuning

- **Confidence threshold** (`CONFIDENCE_THRESHOLD`, default `0.6`): cosine similarity below this triggers the fallback message instead of an LLM call. Raise to be stricter; lower to let the model answer fuzzier questions.
- **Top-K** (`DEFAULT_TOP_K`, default `3`): number of FAQ rows passed to the LLM as context. Both controls are also exposed in the sidebar at runtime.

## Project layout

```
app.py              # Streamlit UI
rag_engine.py       # Embedding, indexing, retrieval, generation
data_loading.py     # FAQ spreadsheet loader
config.py           # Paths, models, prompts, thresholds
faqs/               # Source FAQ spreadsheets (one per department)
vector_stores/      # Persisted FAISS indexes + embedding cache (gitignored)
chat_log/           # Saved chats (gitignored)
tests/              # Smoke tests
```

## Tests

```bash
pip install pytest
pytest
```
