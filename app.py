import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import smtplib
from email.message import EmailMessage

# Chargement de FinBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()

# Fonction d'envoi d'email
def send_email(subject, body, to="paul.pecoraro@essca.eu"):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = "your.email@gmail.com"  # Remplacer par votre email
    msg['To'] = to

    # Connexion sécurisée au serveur SMTP de Gmail
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login("your.email@gmail.com", "your_app_password")  # Remplacer par votre mot de passe d'application
        smtp.send_message(msg)

# Interface utilisateur Streamlit
st.title("Test d'Explicabilité de la Classification de Sentiment FinBERT")

# Zone de saisie de texte
user_input = st.text_area("Entrez un extrait d'actualité financière, un tweet ou un commentaire d'analyste :")

# Exécution de FinBERT
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs[0].numpy()
        labels = ['Négatif', 'Neutre', 'Positif']

    st.subheader("Classification de Sentiment FinBERT")
    for label, prob in zip(labels, probs_np):
        st.write(f"**{label}** : {prob:.2%}")

    # Section Enquête
    st.subheader("Enquête sur l'Explicabilité")
    clarity = st.radio("Quelle est la clarté de la classification de sentiment pour vous ?", 
                       ["Très peu claire", "Peu claire", "Neutre", "Assez claire", "Très claire"])
    explainability = st.radio("Pourriez-vous expliquer ce résultat à un client ou un auditeur ?", 
                            ["Certainement pas", "Probablement pas", "Je ne sais pas", "Probablement oui", "Certainement oui"])
    usefulness = st.radio("Ce résultat vous aide-t-il à prendre une décision d'investissement ?", 
                         ["Pas du tout", "Légèrement", "Modérément", "Fortement", "Extrêmement"])

    # Tâche de Prédiction Inverse
    st.subheader("Tâche de Simulation Inverse")
    user_guess = st.radio("Quel est selon vous le sentiment ?", labels)

    # Tâche de Réécriture
    st.subheader("Défi de Réécriture")
    rewrite_input = st.text_area("Essayez de réécrire la phrase pour changer le sentiment :")

    # Bouton de Soumission
    if st.button("Soumettre l'Enquête"):
        body = f"""
Texte Original : {user_input}
Prédiction de Sentiment FinBERT :
- Négatif : {probs_np[0]:.2%}
- Neutre : {probs_np[1]:.2%}
- Positif : {probs_np[2]:.2%}

Sentiment Prédit par l'Utilisateur : {user_guess}
Texte Réécrit : {rewrite_input}

Réponses de l'Enquête :
- Clarté : {clarity}
- Explicabilité : {explainability}
- Utilité : {usefulness}
"""
        send_email("Nouvelle Réponse à l'Enquête FinBERT", body)
        st.success("Votre réponse a été envoyée au chercheur ! ✅")
