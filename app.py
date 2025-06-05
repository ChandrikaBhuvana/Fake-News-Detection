import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title and description
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is **Real** or **Fake** using a machine learning model trained on real-world data.")

# Text input
news = st.text_area("📝 Paste the news article text here", height=250)

# Prediction button
if st.button("🔍 Predict"):
    if news.strip() == "":
        st.warning("Please enter some news content before clicking Predict.")
    else:
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        if pred[0] == "REAL":
            st.success("✅ This news is **Real**.")
        else:
            st.error("🚫 This news is **Fake**.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and scikit-learn.")
