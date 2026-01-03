import streamlit as st
import joblib
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (for Streamlit Cloud)
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- UI ---------------- #

st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")

st.title("📊 Real-Time Twitter Sentiment Analyzer")
st.write(
    "This application analyzes the sentiment of live tweet text using a "
    "machine learning model trained on a Hugging Face Twitter dataset."
)

user_input = st.text_area("✍️ Enter a tweet or social media post:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        start_time = time.time()

        # Preprocess and predict
        cleaned_text = preprocess_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        latency = time.time() - start_time

        # Handle both numeric and string predictions safely
        if isinstance(prediction, str):
            sentiment = prediction.capitalize()
        else:
            sentiment_map = {
                0: "Negative 😠",
                1: "Neutral 😐",
                2: "Positive 😊"
            }
            sentiment = sentiment_map.get(prediction, "Unknown")

        # Display results
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"⏱️ **Latency:** {latency:.4f} seconds")
