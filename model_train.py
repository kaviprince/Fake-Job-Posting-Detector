import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\Kaviyarasan.K\Documents\Project 2\main\data\cleaned_data.csv")
df["clean_text"] = df["clean_text"].fillna("").astype(str)

X_text = df["clean_text"]
y = df["fraudulent"]
with open(r"C:\Users\Kaviyarasan.K\Documents\Project 2\main\models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
X = vectorizer.transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_acc)

best_model = lr_model if lr_acc >= nb_acc else nb_model

with open(r"C:\Users\Kaviyarasan.K\Documents\Project 2\main\models/fake_job_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(" Best model saved successfully")