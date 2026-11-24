import pandas as pd
import re
import unicodedata
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------
# 1️⃣ Charger datasets joueurs et clubs
# ----------------------------
df_players = pd.read_csv(BASE_DIR / "players.csv", usecols=["name"])
df_clubs = pd.read_csv(BASE_DIR / "clubs.csv", usecols=["name"])

mes_joueurs = df_players["name"].dropna().astype(str).drop_duplicates().tolist()
mes_clubs = df_clubs["name"].dropna().astype(str).drop_duplicates().tolist()

# Use the bundled names directly so entity matching does not depend on native NLP DLLs.
TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)

def normalize_token(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch))

def name_tokens(name):
    return tuple(normalize_token(m.group(0)) for m in TOKEN_RE.finditer(name))

def build_entity_index(names):
    index = {}
    for name in names:
        if isinstance(name, str) and len(name.strip()) > 1:
            parts = name_tokens(name)
            if parts:
                index[parts] = name.strip()
    return index

PLAYER_INDEX = build_entity_index(mes_joueurs)

def build_club_index(names):
    index = {}
    suffixes = (
        ("football", "club"), ("soccer", "club"), ("sporting", "club"),
        ("athletic", "club"), ("football",), ("club",), ("fc",),
        ("afc",), ("cf",), ("spa",), ("gmbh",), ("ag",), ("sa",),
    )
    for name in names:
        if not isinstance(name, str) or len(name.strip()) < 2:
            continue
        raw_parts = [m.group(0) for m in TOKEN_RE.finditer(name)]
        parts = tuple(normalize_token(part) for part in raw_parts)
        if not parts:
            continue
        display_parts = raw_parts[:]
        display = name.strip()
        index[parts] = display
        alias = parts
        changed = True
        while changed and alias:
            changed = False
            for suffix in suffixes:
                if len(alias) > len(suffix) and alias[-len(suffix):] == suffix:
                    alias = alias[:-len(suffix)]
                    display_parts = display_parts[:-len(suffix)]
                    display = " ".join(display_parts)
                    index.setdefault(alias, display)
                    changed = True
                    break
        # Common articles omit legal prefixes such as "FC".
        while len(alias) > 1 and alias[0] in ("fc", "afc", "cf", "sc"):
            alias = alias[1:]
            display_parts = display_parts[1:]
            display = " ".join(display_parts)
            index.setdefault(alias, display)

    # Widely used short forms absent from Transfermarkt's formal club names.
    known_aliases = {
        "man city": ("manchester", "city"),
        "man utd": ("manchester", "united"),
        "man united": ("manchester", "united"),
        "psg": ("paris", "saint", "germain"),
        "spurs": ("tottenham", "hotspur"),
        "barca": ("barcelona",),
        "bayern munich": ("bayern", "munchen"),
    }
    normalized_names = [(name_tokens(name), name) for name in names if isinstance(name, str)]
    for alias, wanted in known_aliases.items():
        canonical = next((name for parts, name in normalized_names
                          if any(parts[i:i + len(wanted)] == wanted
                                 for i in range(len(parts) - len(wanted) + 1))), None)
        if canonical:
            index[name_tokens(alias)] = index.get(wanted, canonical)
    return index

CLUB_INDEX = build_club_index(mes_clubs)

# ----------------------------
# 3️⃣ Fonctions utilitaires
# ----------------------------
def data_cleaning(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>|https?://\S+|www\.\S+', ' ', str(text))
    text = re.sub(r"[-‐‑‒–—]", " ", text)
    text = re.sub(
        r"[^\w€$£\s.,'’]",
        ' ',
        text
    )
    return re.sub(r"\s+", " ", text).strip()

def data_extraction(text):
    results = {
        "joueur": "Non trouvé",
        "club_actuel": "Non trouvé",
        "club_destination": "Non trouvé",
        "prix": "Non mentionné"
    }

    tokens = [normalize_token(m.group(0)) for m in TOKEN_RE.finditer(text)]
    def find_entities(index):
        matches = []
        i = 0
        while i < len(tokens):
            found = None
            for size in range(min(8, len(tokens) - i), 0, -1):
                parts = tuple(tokens[i:i + size])
                display_name = index.get(parts)
                if display_name:
                    found = (i, i + size, display_name)
                    break
            if found:
                matches.append(found)
                i = found[1]
            else:
                i += 1
        return matches

    player_matches = find_entities(PLAYER_INDEX)
    if player_matches:
        results["joueur"] = player_matches[0][2]

    club_matches = find_entities(CLUB_INDEX)
    if club_matches:
        unassigned = []
        for start, end, club_name in club_matches:
            before = tokens[max(0, start - 6):start]
            after = tokens[end:min(len(tokens), end + 4)]
            context = before + after
            if any(word in before for word in ("from", "leaves", "left", "departure", "departing")):
                results["club_actuel"] = club_name
            elif any(word in context for word in ("to", "joins", "join", "joined", "signs", "signing", "signed", "for", "move", "moving")):
                results["club_destination"] = club_name
            else:
                unassigned.append(club_name)
        if len(club_matches) >= 2:
            if results["club_actuel"] == "Non trouvé" and unassigned:
                results["club_actuel"] = unassigned[0]
            if results["club_destination"] == "Non trouvé":
                destination = next((name for name in unassigned
                                    if name != results["club_actuel"]), None)
                if destination:
                    results["club_destination"] = destination
        elif results["club_destination"] == "Non trouvé":
            results["club_destination"] = club_matches[0][2]

    money = re.search(
        r"(?:[€$£]\s*\d+(?:[.,]\d+)?\s*(?:million(?:s)?|billion(?:s)?|bn|m)?|"
        r"\d+(?:[.,]\d+)?\s*(?:million(?:s)?|billion(?:s)?|bn|m|[€$£]))",
        text, re.IGNORECASE
    )
    if money:
        results["prix"] = money.group(0).strip()

    return results

# ----------------------------
# 4️⃣ Modèle TF-IDF + RandomForest (OFFICIEL / RUMEUR)
# ----------------------------
MODEL_PATH = BASE_DIR / "rf_model.pkl"
VECTOR_PATH = BASE_DIR / "vectorizer.pkl"

missing_models = [path.name for path in (MODEL_PATH, VECTOR_PATH) if not path.is_file()]
if missing_models:
    raise FileNotFoundError(
        "Modèle NLP incomplet : fichier(s) manquant(s) "
        + ", ".join(missing_models)
        + ". Replacez rf_model.pkl et vectorizer.pkl dans le dossier du projet."
    )

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTOR_PATH)
if getattr(clf, "n_features_in_", None) != len(vectorizer.get_feature_names_out()):
    raise ValueError(
        "Le modèle et le vectoriseur ne correspondent pas : "
        "leurs nombres de caractéristiques sont différents."
    )
if set(map(str, getattr(clf, "classes_", ()))) != {"OFFICIEL", "RUMEUR"}:
    raise ValueError("Le modèle chargé ne contient pas les classes OFFICIEL et RUMEUR attendues.")

def predict_article_category(text):
    normalized_text = str(text).casefold()
    vec = vectorizer.transform([normalized_text])
    probabilities = clf.predict_proba(vec)[0]
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    model_prediction = clf.classes_[best_index]
    model_confidence = float(max(probabilities))

    # News wording often signals uncertainty more clearly than the old model,
    # which was trained on automatically assigned labels.
    rumor_cues = (
        r"\brumou?rs?\b", r"\b(?:could|may|might) join\b",
        r"\binterested in (?:signing|bringing)\b", r"\blinked with\b",
        r"\breports? (?:suggest|claim|say|link)\b", r"\breportedly\b",
        r"\bno (?:formal )?(?:offer|bid|agreement)\b",
        r"\b(?:considering|exploring) a move\b", r"\bset to (?:join|sign)\b",
        r"\bclose to (?:agree|sign)\w*\b",
    )
    official_cues = (
        r"\bofficially (?:confirmed|announced|completed|unveiled)\b",
        r"\b(?:have|has) confirmed the (?:signing|transfer|deal)\b",
        r"\b(?:have|has) signed (?:a|his|her|their|the)\b",
        r"\bsigned a [\w-]+ contract\b", r"\b(?:has|have) joined\b",
        r"\b(?:deal|transfer) has been completed\b",
        r"\bunveiled as\b", r"\bmedical (?:has been )?(?:completed|passed)\b",
        r"\bfinali[sz]ed (?:the )?(?:deal|agreement|transfer)\b",
    )
    rumor_hits = sum(bool(re.search(pattern, normalized_text, re.IGNORECASE))
                     for pattern in rumor_cues)
    official_hits = sum(bool(re.search(pattern, normalized_text, re.IGNORECASE))
                        for pattern in official_cues)
    officially_negated = re.search(
        r"\b(?:not|never|yet to) (?:been )?(?:officially )?(?:confirmed|announced)\b|"
        r"\bden(?:y|ies|ied) (?:the )?(?:signing|transfer|deal)\b",
        normalized_text, re.IGNORECASE
    )
    if officially_negated:
        official_hits = 0

    if rumor_hits or official_hits:
        if rumor_hits > official_hits:
            prediction = "RUMEUR"
            signal_count = rumor_hits - official_hits
        elif official_hits > rumor_hits:
            prediction = "OFFICIEL"
            signal_count = official_hits - rumor_hits
        else:
            prediction = model_prediction
            signal_count = 1
        method = "Indices textuels + modèle"
        selected_model_probability = float(probabilities[list(clf.classes_).index(prediction)])
        confidence = max(
            selected_model_probability,
            min(0.92, 0.68 + 0.06 * signal_count)
        )
    else:
        prediction = model_prediction
        confidence = model_confidence
        method = "Modèle TF-IDF + Random Forest"
    return prediction, confidence, method

# ----------------------------
# 5️⃣ Fonction finale pour Streamlit
# ----------------------------
def analyze_article(text):
    clean_text = data_cleaning(text)
    entities = data_extraction(clean_text)
    category, confidence, method = predict_article_category(clean_text)
    return {
        "joueur": entities["joueur"],
        "equipe_actuelle": entities["club_actuel"],
        "equipe_cible": entities["club_destination"],
        "prix": entities["prix"],
        "type_accord": category,
        "confiance": confidence,
        "methode_classification": method
    }
