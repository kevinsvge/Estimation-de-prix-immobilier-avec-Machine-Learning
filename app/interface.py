import streamlit as st
from transformers import pipeline

# Configuration de la page
st.set_page_config(page_title="Résumeur IA (GPU)", page_icon="🧠")

st.title("🧠 Résumeur automatique de texte (version GPU avec PyTorch)")
st.markdown("Collez un texte long (article, email, rapport…) et obtenez un résumé généré par l'IA.")

# Chargement du modèle (PyTorch + GPU si dispo)
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt",  # ⚠️ Utilise PyTorch (GPU compatible)
        device=0 if torch.cuda.is_available() else -1
    )

# Import torch pour la gestion GPU
import torch
st.sidebar.markdown(f"🖥️ **GPU disponible** : `{torch.cuda.get_device_name(0)}`" if torch.cuda.is_available() else "⚠️ Aucun GPU détecté.")

summarizer = load_model()

# Entrée utilisateur
text_input = st.text_area("📄 Texte à résumer :", height=250)

# Paramètres réglables
st.sidebar.title("⚙️ Paramètres de résumé")
max_length = st.sidebar.slider("Longueur max du résumé", 50, 300, 130)
min_length = st.sidebar.slider("Longueur min du résumé", 10, 100, 30)

# Résumé
if st.button("✨ Résumer le texte"):
    if text_input.strip() != "":
        with st.spinner("💡 Résumé en cours avec l'accélération GPU..."):
            summary = summarizer(
                text_input,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
        st.subheader("📝 Résumé généré :")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Veuillez saisir un texte pour générer un résumé.")

# Footer
st.markdown("---")
st.caption("Résumé IA propulsé par [🤗 Transformers](https://huggingface.co/facebook/bart-large-cnn) + [PyTorch](https://pytorch.org) – Optimisé GPU")
