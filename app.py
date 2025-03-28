
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load FinBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

# Title
st.title("FinBERT Sentiment Explainability Test")

# Input Text
user_input = st.text_area("Enter a financial news excerpt, tweet, or analyst comment:")

# Run FinBERT
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs[0].numpy()
        labels = ['Negative', 'Neutral', 'Positive']

    # Display Results
    st.subheader("FinBERT Sentiment Classification")
    for label, prob in zip(labels, probs_np):
        st.write(f"**{label}**: {prob:.2%}")

    # Ask Participants to Evaluate Explainability
    st.subheader("Explainability Survey")
    st.radio("How clear is the sentiment classification to you?", ["Very unclear", "Somewhat unclear", "Neutral", "Somewhat clear", "Very clear"])
    st.radio("Could you explain this output to a client or auditor?", ["Definitely not", "Probably not", "Not sure", "Probably yes", "Definitely yes"])
    st.radio("Does the output help inform your investment decision?", ["Not at all", "Slightly", "Moderately", "Strongly", "Extremely"])

    # Reverse Prediction Task
    st.subheader("Reverse Simulation Task")
    user_guess = st.radio("Without seeing the FinBERT output above, what would you guess the sentiment is?", labels)

    # Rewrite Challenge (Counterfactual)
    st.subheader("Rewrite Challenge")
    st.write("Try to rewrite the sentence to make it more **positive** or **negative**, depending on the original tone.")
    rewrite_input = st.text_area("Enter your rewritten version here:")
