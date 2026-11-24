# Football Transfer NLP Analyzer

This project applies Natural Language Processing to football transfer news articles. It extracts key information from a transfer article and classifies the transfer as either official or rumour-based.

## Features

- extract the player mentioned in the article;
- detect the current club and target club when possible;
- extract transfer price information;
- classify the transfer type as `OFFICIEL` or `RUMEUR`;
- display the result in a Streamlit web application;
- use TF-IDF features with a Random Forest classifier;
- match player names and common club aliases from the bundled CSV files;
- combine the Random Forest prediction with explicit wording cues for confirmed deals and rumours.

## Project Structure

```text
app.py                         Streamlit application
projet_nlp.py                  NLP pipeline and prediction logic
projet_nlp.ipynb               Notebook demonstrating the current pipeline
90minFootballTransferNewsNLP.csv  Source corpus (not loaded by the app)
players.csv                    Player names dataset
clubs.csv                      Club names dataset
football_db_clean.csv          Optional prepared entity export
rf_model.pkl                   Trained Random Forest model
vectorizer.pkl                 TF-IDF vectorizer
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The app requires both trained artifacts, `rf_model.pkl` and `vectorizer.pkl`.
It stops with a clear error if either file is missing instead of silently
training a replacement from automatically guessed labels.

## Run the Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal. The tested package
versions are pinned in `requirements.txt` to keep the saved model compatible.

## How Recognition Works

Player and club matching is accent-insensitive. Club names are matched against both
their formal names and common short forms (for example, `Manchester City` and
`Man City`). Nearby transfer phrases such as `from`, `joins`, and `signing for`
help assign the current and destination clubs. Transfer prices are extracted
with currency and million/billion patterns.

The category combines the saved TF-IDF/Random Forest model with explicit text
cues such as `officially confirmed`, `could join`, and `no offer`. The displayed
score is indicative; it is not a calibrated accuracy or probability. Entity
recognition is limited to names and aliases represented by the bundled data and
the aliases defined in `projet_nlp.py`.

## Example Use Case

Paste a football transfer article into the text area. The application returns:

- detected player;
- current club;
- target club;
- transfer price;
- agreement type;
- indicative classification score and method.

## Technologies

```text
Python | Streamlit | scikit-learn | TF-IDF | Random Forest | pandas
```

