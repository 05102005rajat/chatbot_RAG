import os
import streamlit as st
from datetime import datetime
from rag_engine import (
    embed_with_ollama, create_faiss_index, save_faiss_index,
    load_faiss_index, search_index, generate_rag_answer
)
from data_loading import load_faq_data
from config import PREDEFINED_RESPONSES
from config import (
    FAQ_FOLDER,
    INDEX_FOLDER,
    CHAT_LOG_FOLDER,
    EMBEDDING_MODEL,
    RAG_MODEL,
    DEPARTMENT_FILES,
    PREDEFINED_RESPONSES
)


# --- Init session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current_questions" not in st.session_state:
    st.session_state.current_questions = None
if "current_answers" not in st.session_state:
    st.session_state.current_answers = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None

# --- Folder setup ---

os.makedirs(CHAT_LOG_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)


# --- Chat log handling ---
def list_chat_files():
    files = [f for f in os.listdir(CHAT_LOG_FOLDER) if f.endswith(".txt")]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(CHAT_LOG_FOLDER, f)), reverse=True)

def save_chat_history(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CHAT_LOG_FOLDER}/{st.session_state.selected_department}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for speaker, msg in history:
            f.write(f"{speaker}: {msg}\n")
    return filename

def load_chat_file(filename):
    chat = []
    with open(os.path.join(CHAT_LOG_FOLDER, filename), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("You:"):
                chat.append(("You", line[4:].strip()))
            elif line.startswith("Bot:"):
                chat.append(("Bot", line[4:].strip()))
    return chat

def load_department_data(department):
    faq_file = DEPARTMENT_FILES[department]
    faq_path = os.path.join(FAQ_FOLDER, faq_file)
    index_path = os.path.join(INDEX_FOLDER, f"{department.lower()}_faq.index")
    
    # Load FAQs
    questions, answers = load_faq_data(faq_path)
    
    # Load or create index
    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
    else:
        with st.spinner("Creating department index..."):
            embeddings = embed_with_ollama(questions)
            index = create_faiss_index(embeddings)
            save_faiss_index(index, index_path)
    
    return questions, answers, index

# --- Sidebar UI ---
st.sidebar.title("Settings")
department_options = list(DEPARTMENT_FILES.keys())

if "selected_department" not in st.session_state:
    st.session_state.selected_department = department_options[0]
    st.session_state.current_questions, st.session_state.current_answers, st.session_state.current_index = load_department_data(department_options[0])

selected = st.sidebar.selectbox(
    "Choose Department",
    department_options,
    index=department_options.index(st.session_state.selected_department)
)

if selected != st.session_state.selected_department:
    st.session_state.selected_department = selected
    st.session_state.history = []
    st.session_state.current_questions, st.session_state.current_answers, st.session_state.current_index = load_department_data(selected)
    st.rerun()

top_k = st.sidebar.slider("Number of Answers to Use for Context", 1, 5, 3)
clear_cache = st.sidebar.button("Clear Index Cache")
clear_chat = st.sidebar.button("Clear Chat History")

# --- Chat controls ---
chat_files = list_chat_files()
selected_file = st.sidebar.selectbox("View Previous Chat", chat_files if chat_files else ["No chats found"])

if st.sidebar.button("Load Selected Chat") and selected_file != "No chats found":
    st.session_state.history = load_chat_file(selected_file)
    st.rerun()

if st.sidebar.button("Save Chat"):
    if st.session_state.history:
        filename = save_chat_history(st.session_state.history)
        st.sidebar.success(f"Chat saved to {filename}")

if st.sidebar.button("Start New Chat"):
    if st.session_state.history:
        save_chat_history(st.session_state.history)
    st.session_state.history = []
    st.rerun()

# Handle clear operations
if clear_cache:
    index_path = os.path.join(INDEX_FOLDER, f"{st.session_state.selected_department.lower()}_faq.index")
    if os.path.exists(index_path):
        os.remove(index_path)
        st.session_state.current_questions, st.session_state.current_answers, st.session_state.current_index = load_department_data(st.session_state.selected_department)
        st.sidebar.success("Index cache cleared and rebuilt!")
        st.rerun()

if clear_chat:
    st.session_state.history = []
    st.rerun()

# --- Main Chat UI ---
st.title("🧠 RAG-powered FAQ Chatbot")
st.write(f"Current Department: {st.session_state.selected_department}")

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.chat_message("user").markdown(f"**🧍 You:** {msg}")
    else:
        st.chat_message("assistant").markdown(f"**🤖 Bot:** {msg}")

user_input = st.chat_input("Ask me anything:", key="chat_input")

if user_input:
    user_input_clean = user_input.strip().lower()

    if user_input_clean in PREDEFINED_RESPONSES:
        answer = PREDEFINED_RESPONSES[user_input_clean]
    else:
        with st.spinner("Thinking..."):
            answer = generate_rag_answer(
                user_input, 
                st.session_state.current_index,
                st.session_state.current_questions,
                st.session_state.current_answers,
                top_k
            )

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))
    st.rerun()