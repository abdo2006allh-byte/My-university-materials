import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\Abdullah\Desktop\project NLP\data& preprocessing\tourism_dataset.csv")

df['search_text'] = df.apply(lambda x: f"{x['Category']} tourism in {x['Country']} at {x['Location']}. Rating: {x['Rating']}", axis=1)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

print("Processing Data...")
data_embeddings = get_embeddings(df['search_text'].head(100).tolist())

def get_chatbot_response(user_query):
    query_embedding = get_embeddings([user_query])
    similarities = cosine_similarity(query_embedding, data_embeddings)
    best_index = similarities.argmax()
    
    row = df.iloc[best_index]
    response = f"I found a great place! It's a {row['Category']} spot in {row['Country']} located at {row['Location']}. It has a rating of {row['Rating']}."
    return response