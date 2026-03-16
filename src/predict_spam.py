# src/predict_spam.py

import joblib
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Load saved model and TF-IDF vectorizer
model = joblib.load("../models/spam_model.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

# Prepare stopwords
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Function to predict spam
def predict_spam(text):
    clean_text_input = clean_text(text)
    vector = tfidf.transform([clean_text_input])
    return model.predict(vector)[0]

# Example usage
if __name__ == "__main__":
    while True:
        msg = input("Enter a message to check (or 'exit' to quit): ")
        if msg.lower() == 'exit':
            break
        result = predict_spam(msg)
        print(f"Result: {result}\n")