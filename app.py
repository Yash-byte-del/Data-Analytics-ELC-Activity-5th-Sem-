import streamlit as st
import joblib
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# UI
st.title("Real-Time Twitter Sentiment Analyzer")
st.write("Enter a tweet to analyze its sentiment in real time.")

user_input = st.text_area("Tweet Text")

if st.button("Analyze Sentiment"):
    start_time = time.time()
    
    cleaned = preprocess_text(user_input)
    vectorized = tfidf.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]  # 👈 FIX HERE
    
    latency = time.time() - start_time

    sentiment_map = {
        0: "Negative 😠",
        1: "Neutral 😐",
        2: "Positive 😊"
    }

    st.success(f"Predicted Sentiment: {sentiment_map[int(prediction)]}")
    st.write(f"Latency: {latency:.4f} seconds")
