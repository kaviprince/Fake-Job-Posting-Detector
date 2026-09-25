import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv(r"C:\Users\Kaviyarasan.K\Documents\Project 2\main\data\cleaned_data.csv")

vectorizer = TfidfVectorizer(max_features=5000)
df['clean_text'] = df['clean_text'].fillna("")

X = vectorizer.fit_transform(df['clean_text'])
y = df['fraudulent']
with open( r"C:\Users\Kaviyarasan.K\Documents\Project 2\main\models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)