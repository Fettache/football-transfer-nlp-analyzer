# Football Transfer NLP Analyzer

This project applies Natural Language Processing to football transfer news articles. It extracts key information from a transfer article and classifies the transfer as either official or rumour-based.

## Features

- extract the player mentioned in the article;
- detect the current club and target club when possible;
- extract transfer price information;
- classify the transfer type as `OFFICIEL` or `RUMEUR`;
- display the result in a Streamlit web application;
- use TF-IDF features with a Random Forest classifier;
- use spaCy and an `EntityRuler` for player and club recognition.

## Project Structure

```text
app.py                         Streamlit application
projet_nlp.py                  NLP pipeline and prediction logic
projet_nlp.ipynb               Notebook used during experimentation
90minFootballTransferNewsNLP.csv  Football transfer articles
players.csv                    Player names dataset
clubs.csv                      Club names dataset
football_db_clean.csv          Prepared football entities
rf_model.pkl                   Trained Random Forest model
vectorizer.pkl                 TF-IDF vectorizer
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Example Use Case

Paste a football transfer article into the text area. The application returns:

- detected player;
- current club;
- target club;
- transfer price;
- agreement type;
- model confidence.

## Technologies

```text
Python | Streamlit | spaCy | scikit-learn | TF-IDF | Random Forest | pandas
```

