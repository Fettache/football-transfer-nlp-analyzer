import pandas as pd
import numpy as np
import spacy
from spacy.pipeline import EntityRuler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib
import os

# ----------------------------
# 1️⃣ Charger datasets joueurs et clubs
# ----------------------------
df_players = pd.read_csv("players.csv")
df_clubs = pd.read_csv("clubs.csv")

df_players = df_players[['name']].rename(columns={'name': 'Name'})
df_players['Type'] = 'JOUEUR'
df_clubs = df_clubs[['name']].rename(columns={'name': 'Name'})
df_clubs['Type'] = 'CLUB'

football_db = pd.concat([df_players, df_clubs], ignore_index=True)
football_db.dropna(inplace=True)
football_db.drop_duplicates(subset=['Name'], inplace=True)

mes_joueurs = football_db[football_db['Type'] == 'JOUEUR']['Name'].tolist()
mes_clubs = football_db[football_db['Type'] == 'CLUB']['Name'].tolist()

# ----------------------------
# 2️⃣ Initialisation SpaCy + EntityRuler
# ----------------------------
nlp = spacy.load("en_core_web_sm")

def setup_model(nlp, joueurs_list, clubs_list):
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    patterns = []

    # Joueurs
    for name in joueurs_list:
        if isinstance(name, str) and len(name) > 2:
            patterns.append({
                "label": "PERSON",
                "pattern": [{"LOWER": part.lower()} for part in name.split()]
            })

    # Clubs
    for club in clubs_list:
        if isinstance(club, str) and len(club) > 1:
            parts = club.split()
            patterns.append({
                "label": "CLUB",
                "pattern": [{"LOWER": p.lower()} for p in parts]
            })
            if len(parts) > 1:
                patterns.append({
                    "label": "CLUB",
                    "pattern": [{"LOWER": parts[0].lower()}]
                })

    # Prix, contrats
    extra_patterns = [
        {"label": "MONEY", "pattern": [{"IS_DIGIT": True},
         {"LOWER": {"REGEX": "^(m|million|m€|m\\$|m£|€|\\$|£)$"}}]},
        {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": "^[€\\$£]"}},
         {"IS_DIGIT": True}, {"LOWER": "m", "OP": "?"}]},
        {"label": "CLUB", "pattern": [{"LOWER": "psg"}]},
        {"label": "CLUB", "pattern": [{"LOWER": "om"}]},
        {"label": "CLUB", "pattern": [{"LOWER": "ol"}]},
        {"label": "CLUB", "pattern": [{"LOWER": "man"}, {"LOWER": "utd"}]},
    ]

    patterns.extend(extra_patterns)
    ruler.add_patterns(patterns)
    return nlp

nlp = setup_model(nlp, mes_joueurs, mes_clubs)

# ----------------------------
# 3️⃣ Fonctions utilitaires
# ----------------------------
def data_cleaning(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>|http\S+', '', str(text))
    text = re.sub(
        r'[^a-zA-Z0-9€$£\s.,\'àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]',
        '',
        text
    )
    return text.strip()

def data_extraction(text):
    doc = nlp(text)
    results = {
        "joueur": "Non trouvé",
        "club_actuel": "Non trouvé",
        "club_destination": "Non trouvé",
        "prix": "Non mentionné",
        "type_contrat": "Non mentionné"
    }

    clubs_detectes = [ent for ent in doc.ents if ent.label_ in ["CLUB", "ORG", "GPE"]]

    for ent in doc.ents:
        if ent.label_ == "PERSON" and results["joueur"] == "Non trouvé":
            results["joueur"] = ent.text
        elif ent.label_ == "MONEY":
            results["prix"] = ent.text

    if len(clubs_detectes) == 1:
        results["club_destination"] = clubs_detectes[0].text
    elif len(clubs_detectes) >= 2:
        for club in clubs_detectes:
            start_idx = max(0, club.start - 3)
            contexte = [t.lower_ for t in doc[start_idx:club.start] if not t.is_punct]
            if any(m in contexte for m in ["from", "leaves", "of", "departure"]):
                results["club_actuel"] = club.text
            elif any(m in contexte for m in ["to", "joins", "signs", "for", "signing"]):
                results["club_destination"] = club.text

    if results["club_destination"] == "Non trouvé" and len(clubs_detectes) > 0:
        for c in clubs_detectes:
            if c.text != results["club_actuel"]:
                results["club_destination"] = c.text
                break

    return results

# ----------------------------
# 4️⃣ Modèle TF-IDF + RandomForest (OFFICIEL / RUMEUR)
# ----------------------------
MODEL_PATH = "rf_model.pkl"
VECTOR_PATH = "vectorizer.pkl"
CSV_PATH = "90minFootballTransferNewsNLP.csv"

df = pd.read_csv(CSV_PATH)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df.dropna(subset=['Content'], inplace=True)

nlp_lemma = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

def lemmatization(text):
    doc = nlp_lemma(str(text).lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

def auto_labeling(row):
    text = (str(row['Title']) + " " + str(row['Content'])).lower()
    text_lem = lemmatization(text)

    official_words = [
        "finalised", "confirmed", "official", "signed", "unveiled",
        "here we go", "agreed a fee", "medical tests",
        "done", "set to", "join", "agree"
    ]

    if any(word in text_lem for word in official_words):
        return "OFFICIEL"
    return "RUMEUR"

if 'category' not in df.columns:
    df['category'] = df.apply(auto_labeling, axis=1)

# Équilibrage (2 classes uniquement)
n_cible = 2000
df_r = df[df.category == "RUMEUR"]
df_o = df[df.category == "OFFICIEL"]

df_balanced = pd.concat([
    resample(
        df_r,
        n_samples=min(n_cible, len(df_r)),
        replace=len(df_r) < n_cible,
        random_state=42
    ),
    resample(
        df_o,
        n_samples=min(n_cible, len(df_o)),
        replace=len(df_o) < n_cible,
        random_state=42
    )
])


df_balanced['Features'] = df_balanced['Title'] + " " + df_balanced['Content'].apply(lemmatization)

X = df_balanced['Features'].astype(str)
y = df_balanced['category']

if os.path.exists(MODEL_PATH) and os.path.exists(VECTOR_PATH):
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTOR_PATH)
else:
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2)
    )
    X_vec = vectorizer.fit_transform(X)
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_vec, y)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTOR_PATH)

def predict_article_category(text):
    text_lem = lemmatization(text)
    vec = vectorizer.transform([text_lem])
    prediction = clf.predict(vec)[0]
    proba = clf.predict_proba(vec).max()
    return prediction, proba

# ----------------------------
# 5️⃣ Fonction finale pour Streamlit
# ----------------------------
def analyze_article(text):
    clean_text = data_cleaning(text)
    entities = data_extraction(clean_text)
    category, confidence = predict_article_category(clean_text)
    return {
        "joueur": entities["joueur"],
        "equipe_actuelle": entities["club_actuel"],
        "equipe_cible": entities["club_destination"],
        "prix": entities["prix"],
        "type_accord": category,
        "confiance": confidence
    }
