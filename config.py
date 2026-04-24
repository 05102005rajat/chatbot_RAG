# === Folder Paths ===
FAQ_FOLDER = "faqs"
INDEX_FOLDER = "vector_stores"
EMBEDDING_CACHE_FOLDER = "vector_stores/_embeddings"
CHAT_LOG_FOLDER = "chat_log"

# === Model Configuration ===
EMBEDDING_MODEL = "embeddinggemma:latest"
RAG_MODEL = "gemma3:1b"

# === Retrieval Configuration ===
# Minimum cosine similarity for the top match before we trust the context
# enough to ask the LLM to answer. Below this, we return a polite fallback
# instead of risking a hallucination.
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_TOP_K = 3

# === Department Mapping ===
DEPARTMENT_FILES = {
    "IT": "IT_FAQs.xlsx",
    "HR": "HR_FAQs.xlsx",
    "Admin": "Admin_FAQs.xlsx",
    "Finance": "Finance_FAQs.xlsx",
    "Security": "Security_FAQs.xlsx",
    "L&D": "L&D_FAQs.xlsx",
    "CSR": "CSR_FAQs.xlsx",
    "Travel": "Travel_FAQs.xlsx",
    "Cafeteria": "Cafeteria_FAQs.xlsx",
    "IT Assets": "IT_Assets_FAQs.xlsx",
}

# === System Prompt ===
SYSTEM_PROMPT = (
    "You are an FAQ assistant for the {department} department. "
    "Answer the user's question using ONLY the context below. "
    "If the context does not contain the answer, say "
    "\"I don't have information on that — please contact the {department} team directly.\" "
    "Do not invent policies, names, dates, or numbers. Be concise."
)

CHITCHAT_PROMPT = (
    "You are a friendly assistant for the {department} team's FAQ chatbot. "
    "The user's message did not match any FAQ. "
    "If it's a greeting, small talk, or thanks, reply warmly and briefly "
    "and invite them to ask a {department}-related question. "
    "If it looks like a real question you don't have FAQ context for, "
    "say you don't have information on that and suggest contacting the "
    "{department} team directly. Do not invent policies, names, or numbers."
)
