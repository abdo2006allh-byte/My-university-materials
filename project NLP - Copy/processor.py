# write in terminal
#   cd "C:\Users\Abdullah\Desktop\project NLP"
#    pip install scikit-learn pandas streamlit
#     streamlit run app.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

csv_path = r"C:\Users\Abdullah\Desktop\project NLP\data& preprocessing\tourism_dataset.csv"

def load_data():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.head(5000).copy()

df = load_data()
df['search_text'] = df.apply(lambda x: f"{x['Category']} in {x['Country']} at {x['Location']}", axis=1)

vectorizer = TfidfVectorizer()
data_matrix = vectorizer.fit_transform(df['search_text'])

def get_chatbot_response(user_query):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, data_matrix)
    best_match = similarities.argmax()
    row = df.iloc[best_match]
    return f"I suggest {row['Location']} in {row['Country']} ({row['Category']}). Rating: {row['Rating']}"