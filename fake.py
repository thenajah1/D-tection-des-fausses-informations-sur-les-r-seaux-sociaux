import streamlit as st
import torch
from transformers import BertTokenizerFast
from model import BERT_Arch  # Assurez-vous d'avoir le modèle défini dans un fichier séparé

# Charger le modèle et le tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch(model.py)  # Remplacez 'bert' par le modèle chargé

# Interface utilisateur
st.title("Application de détection de fausses informations")
text_input = st.text_area("Entrez le texte de l'article de presse", "")

if st.button("Analyser"):
    # Prétraitement du texte saisi
    processed_text = preprocess(text_input)  # Remplacez par votre fonction de prétraitement

    # Utiliser le modèle pour la prédiction
    with torch.no_grad():
        input_ids = tokenizer.encode(processed_text, add_special_tokens=True)
        logits = model(torch.tensor([input_ids]))

    # Afficher les résultats
    prediction = torch.argmax(logits).item()
    if prediction == 0:
        st.write("L'article est prédit comme étant faux.")
    else:
        st.write("L'article est prédit comme étant vrai.")
