import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# Streamlit UI
st.title("Fake News Detector")
news_text = st.text_area("Enter news text:")

if st.button("Check"):
    processed_text = preprocess_text(news_text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    result = "Fake" if prediction == 1 else "Real"
    st.write(f"### Prediction: {result}")
