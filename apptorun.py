import streamlit as st
import pickle
import re

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# -------------------------------
# UI Design
# -------------------------------
st.set_page_config(page_title="Fake Job Detector", layout="centered")

st.title("🕵️ Fake Job Posting Detector")
st.write("Enter a job description to check if it's **Real or Fake**")

# Input box
user_input = st.text_area("Job Description", height=200)

# Button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        # Preprocess
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vector)[0]

        # Output
        if prediction == 1:
            st.error("🚨 This looks like a FAKE job posting")
        else:
            st.success("✅ This looks like a REAL job posting")