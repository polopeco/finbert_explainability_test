import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import smtplib
from email.message import EmailMessage

# 📦 Charger les identifiants de l'email depuis st.secrets
EMAIL_ADDRESS = st.secrets["EMAIL_USER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASS"]

# 📥 Charger FinBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

# ✉️ Fonction d'envoi d'email
def send_email(subject, body, to="pauljean.pecoraro@gmail.com"):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'email : {e}")

# 🎛️ Interface Streamlit
st.title("Test d'Explicabilité du Modèle FinBERT")

# 📝 Texte d'entrée
user_input = st.text_area("Entrez un extrait d'actualité financière, un tweet ou un commentaire d'analyste :")

# 📊 Exécution de FinBERT
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs[0].numpy()
        labels = ['Négatif', 'Neutre', 'Positif']

    st.subheader("Classification du Sentiment par FinBERT")
    for label, prob in zip(labels, probs_np):
        st.write(f"**{label}** : {prob:.2%}")

    # 🧠 Questionnaire
    st.subheader("Questionnaire sur l'Explicabilité")
    clarity = st.radio("Dans quelle mesure la classification du sentiment vous semble-t-elle claire ?", ["Très peu claire", "Plutôt peu claire", "Neutre", "Plutôt claire", "Très claire"])
    explainability = st.radio("Pourriez-vous expliquer ce résultat à un client ou un auditeur ?", ["Certainement pas", "Probablement pas", "Je ne sais pas", "Probablement oui", "Certainement oui"])
    usefulness = st.radio("Ce résultat vous aide-t-il à prendre une décision d'investissement ?", ["Pas du tout", "Un peu", "Moyennement", "Beaucoup", "Énormément"])

    # 🔄 Simulation inverse
    st.subheader("Simulation Inverse")
    user_guess = st.radio("Quel sentiment pensez-vous que le modèle a identifié ?", labels)

    # ✍️ Réécriture
    st.subheader("Défi de Réécriture")
    rewrite_input = st.text_area("Essayez de réécrire la phrase pour changer le sentiment :")

    if rewrite_input.strip():
        rewrite_inputs = tokenizer(rewrite_input, return_tensors="pt", truncation=True)
        with torch.no_grad():
            rewrite_outputs = model(**rewrite_inputs)
            rewrite_probs = torch.nn.functional.softmax(rewrite_outputs.logits, dim=-1)
            rewrite_probs_np = rewrite_probs[0].numpy()

        st.markdown("**Analyse de Sentiment du Texte Réécrit :**")
        for label, prob in zip(labels, rewrite_probs_np):
            st.write(f"**{label}** : {prob:.2%}")

    # 📬 Envoi des réponses
    if st.button("Soumettre le Questionnaire"):
        body = f"""
        Texte original : {user_input}
        Prédiction FinBERT :
        - Négatif : {probs_np[0]:.2%}
        - Neutre : {probs_np[1]:.2%}
        - Positif : {probs_np[2]:.2%}

        Sentiment prédit par l'utilisateur : {user_guess}
        Texte réécrit : {rewrite_input}

        Réponses au questionnaire :
        - Clarté : {clarity}
        - Explicabilité : {explainability}
        - Utilité : {usefulness}
        """
        send_email("Nouvelle réponse au questionnaire FinBERT", body)
        st.success("Votre réponse a bien été envoyée au chercheur ! ✅")
