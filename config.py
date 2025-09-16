FAQ_DATA_PATH = "faqs/IT_FAQs.xlsx"  # Default path, not used for multi-department
INDEX_PATH = "vector_stores/it_faq.index"  # Default path, not used for multi-department
EMBEDDING_MODEL = "embeddinggemma:latest"
RAG_MODEL = "gemma3:1b"

PREDEFINED_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How’s your day going?",
    "hey": "Hey! What can I do for you?",
    "thank you": "You’re welcome! Always happy to help.",
    "thanks": "No problem at all!",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "See you later! Take care."
}