FAQ_DATA_PATH = "faqs/IT_FAQs.xlsx"  # Default path, not used for multi-department
INDEX_PATH = "vector_stores/it_faq.index"  # Default path, not used for multi-department
EMBEDDING_MODEL = "embeddinggemma:latest"
RAG_MODEL = "gemma3:1b"

# config.py

# === Folder Paths ===
FAQ_FOLDER = "faqs"
INDEX_FOLDER = "vector_stores"
CHAT_LOG_FOLDER = "chat_log"

# === Model Configuration ===
EMBEDDING_MODEL = "embeddinggemma:latest"
RAG_MODEL = "gemma3:1b"

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
    "IT Assets": "IT_Assets_FAQs.xlsx"
}

# === Predefined Responses ===
PREDEFINED_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How’s your day going?",
    "hey": "Hey! What can I do for you?",
    "thank you": "You’re welcome! Always happy to help.",
    "thanks": "No problem at all!",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "See you later! Take care."
}
