# app.py
import streamlit as st
from math import isfinite
from numbers import Real

try:
    from projet_nlp import analyze_article
    pipeline_error = None
except Exception as exc:
    analyze_article = None
    pipeline_error = exc

st.set_page_config(page_title="Football Transfers NLP", layout="wide")

st.title("Analyseur de Transferts de Football ⚽")

if pipeline_error is not None:
    st.error("Le pipeline NLP n'a pas pu être chargé. Vérifiez les modèles et les fichiers de données du projet.")
    st.exception(pipeline_error)
    st.stop()

article_input = st.text_area(
    "Collez ici l'article de transfert :",
    height=200,
    max_chars=30000,
    placeholder="Collez un article en anglais sur un transfert de football…"
)

if st.button("Analyser l'article"):
    if article_input.strip() == "":
        st.warning("Veuillez entrer un texte !")
    else:
        try:
            result = analyze_article(article_input)
            st.subheader("Résultat de l'analyse :")
            st.write(f"**Joueur détecté :** {result.get('joueur', 'Non trouvé')}")
            st.write(f"**Équipe actuelle :** {result.get('equipe_actuelle', 'Non trouvé')}")
            st.write(f"**Équipe cible :** {result.get('equipe_cible', 'Non trouvé')}")
            st.write(f"**Prix du transfert :** {result.get('prix', 'Non mentionné')}")
            st.write(f"**Type d'accord :** {result.get('type_accord', 'Non déterminé')}")
            confidence = result.get('confiance')
            if isinstance(confidence, Real) and isfinite(float(confidence)):
                st.write(f"**Score indicatif :** {float(confidence):.0%}")
            else:
                st.write("**Score indicatif :** indisponible")
            method = result.get('methode_classification') or "Modèle TF-IDF + Random Forest"
            st.caption(f"Classification : {method}")
        except Exception as exc:
            st.error("L'analyse a échoué pour cet article.")
            st.exception(exc)
st.markdown("---")
st.subheader("📈 Performances du modèle")

st.info("""
Le classement combine un modèle TF-IDF + Random Forest et des indices textuels explicites
(par exemple « officially confirmed » ou « could join »). Le score est indicatif et n'est
pas une mesure de précision calibrée. Les équipes et les joueurs sont reconnus à partir
des fichiers de noms fournis avec le projet.
""")
