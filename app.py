import os
import json
import re
from datetime import datetime

import streamlit as st

from rag_engine import (
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    get_or_build_embeddings,
    retrieve,
    is_confident,
    stream_answer,
    fallback_message,
    OllamaUnavailable,
)
from data_loading import load_faq_data
from config import (
    FAQ_FOLDER,
    INDEX_FOLDER,
    CHAT_LOG_FOLDER,
    DEPARTMENT_FILES,
    DEFAULT_TOP_K,
    CONFIDENCE_THRESHOLD,
)


# --- Session state ---
st.session_state.setdefault("history", [])
st.session_state.setdefault("current_questions", None)
st.session_state.setdefault("current_answers", None)
st.session_state.setdefault("current_index", None)

os.makedirs(CHAT_LOG_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)


# --- Chat log handling (JSON, with .txt back-compat for old logs) ---
def list_chat_files():
    files = [
        f for f in os.listdir(CHAT_LOG_FOLDER) if f.endswith((".json", ".txt"))
    ]
    return sorted(
        files,
        key=lambda f: os.path.getmtime(os.path.join(CHAT_LOG_FOLDER, f)),
        reverse=True,
    )


def save_chat_history(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CHAT_LOG_FOLDER}/{st.session_state.selected_department}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "department": st.session_state.selected_department,
                "saved_at": timestamp,
                "messages": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return filename


def load_chat_file(filename):
    path = os.path.join(CHAT_LOG_FOLDER, filename)
    if filename.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("messages", [])
    # Legacy .txt fallback — old line-prefixed format.
    chat = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("You:"):
                chat.append({"role": "user", "content": line[4:].strip()})
            elif line.startswith("Bot:"):
                chat.append({"role": "assistant", "content": line[4:].strip()})
    return chat


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def index_path_for(department: str) -> str:
    return os.path.join(INDEX_FOLDER, f"{_slug(department)}_faq.index")


def load_department_data(department):
    faq_file = DEPARTMENT_FILES[department]
    faq_path = os.path.join(FAQ_FOLDER, faq_file)
    index_path = index_path_for(department)

    questions, answers = load_faq_data(faq_path)

    if os.path.exists(index_path):
        index = load_faiss_index(index_path)
    else:
        with st.spinner(f"Indexing {department} FAQs (one-time)..."):
            try:
                embeddings = get_or_build_embeddings(questions, faq_path)
            except OllamaUnavailable as e:
                st.error(str(e))
                st.stop()
            index = create_faiss_index(embeddings)
            save_faiss_index(index, index_path)
    return questions, answers, index


def render_sources(sources, questions, answers):
    if not sources:
        return
    label = f"Sources ({len(sources)}) — top match {sources[0][1]:.2f}"
    with st.expander(label):
        for idx, score in sources:
            if not (0 <= idx < len(questions)):
                continue
            st.markdown(f"**[{score:.2f}] Q:** {questions[idx]}")
            st.markdown(f"**A:** {answers[idx]}")
            st.divider()


# --- Sidebar ---
st.sidebar.title("Settings")
department_options = list(DEPARTMENT_FILES.keys())

if "selected_department" not in st.session_state:
    st.session_state.selected_department = department_options[0]
    (
        st.session_state.current_questions,
        st.session_state.current_answers,
        st.session_state.current_index,
    ) = load_department_data(department_options[0])

selected = st.sidebar.selectbox(
    "Choose Department",
    department_options,
    index=department_options.index(st.session_state.selected_department),
)

if selected != st.session_state.selected_department:
    st.session_state.selected_department = selected
    st.session_state.history = []
    (
        st.session_state.current_questions,
        st.session_state.current_answers,
        st.session_state.current_index,
    ) = load_department_data(selected)
    st.rerun()

top_k = st.sidebar.slider("Context size (top_k)", 1, 5, DEFAULT_TOP_K)
threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
)

clear_cache = st.sidebar.button("Clear Index Cache")
clear_chat = st.sidebar.button("Clear Chat History")

chat_files = list_chat_files()
selected_file = st.sidebar.selectbox(
    "View Previous Chat", chat_files if chat_files else ["No chats found"]
)

if st.sidebar.button("Load Selected Chat") and selected_file != "No chats found":
    st.session_state.history = load_chat_file(selected_file)
    st.rerun()

if st.sidebar.button("Save Chat"):
    if st.session_state.history:
        filename = save_chat_history(st.session_state.history)
        st.sidebar.success(f"Saved to {filename}")

if st.sidebar.button("Start New Chat"):
    if st.session_state.history:
        save_chat_history(st.session_state.history)
    st.session_state.history = []
    st.rerun()

if clear_cache:
    index_path = index_path_for(st.session_state.selected_department)
    if os.path.exists(index_path):
        os.remove(index_path)
    (
        st.session_state.current_questions,
        st.session_state.current_answers,
        st.session_state.current_index,
    ) = load_department_data(st.session_state.selected_department)
    st.sidebar.success("Index rebuilt.")
    st.rerun()

if clear_chat:
    st.session_state.history = []
    st.rerun()


# --- Main chat ---
st.title("RAG-powered FAQ Chatbot")
st.caption(f"Current department: **{st.session_state.selected_department}**")

questions = st.session_state.current_questions
answers = st.session_state.current_answers
index = st.session_state.current_index

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"), questions, answers)

user_input = st.chat_input("Ask me anything:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            relevant_idxs, similarities = retrieve(user_input, index, top_k)
        except OllamaUnavailable as e:
            st.error(str(e))
            st.stop()

        sources = list(zip(relevant_idxs, similarities))

        if not is_confident(similarities, threshold):
            answer = fallback_message(st.session_state.selected_department)
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(
                    stream_answer(
                        user_input,
                        questions,
                        answers,
                        relevant_idxs,
                        st.session_state.selected_department,
                    )
                )
            except OllamaUnavailable as e:
                st.error(str(e))
                st.stop()

        render_sources(sources, questions, answers)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
