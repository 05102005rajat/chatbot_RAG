import pandas as pd

def load_faq_data(faq_path):
    df = pd.read_excel(faq_path)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    questions = df['question'].astype(str).tolist()
    answers = df['answer'].astype(str).tolist()
    return questions, answers