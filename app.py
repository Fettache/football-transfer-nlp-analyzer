# app.py
import streamlit as st
from projet_nlp import analyze_article

st.set_page_config(page_title="Football Transfers NLP", layout="wide")

st.title("Analyseur de Transferts de Football ⚽")

article_input = st.text_area("Collez ici l'article de transfert :", height=200)

if st.button("Analyser l'article"):
    if article_input.strip() == "":
        st.warning("Veuillez entrer un texte !")
    else:
        result = analyze_article(article_input)
        st.subheader("Résultat de l'analyse :")
        st.write(f"**Joueur détecté :** {result['joueur']}")
        st.write(f"**Équipe actuelle :** {result['equipe_actuelle']}")
        st.write(f"**Équipe cible :** {result['equipe_cible']}")
        st.write(f"**Prix du transfert :** {result['prix']}")
        st.write(f"**Type d'accord :** {result['type_accord']}")
        st.write(f"**Confiance du modèle :** {result['confiance']:.0%}")
st.markdown("---")
st.subheader("📈 Performances du modèle")

st.success("""
Accuracy ≈ 0.87  
Random Forest – TF-IDF (1–2 grams)  
Classes : OFFICIEL / RUMEUR
""")