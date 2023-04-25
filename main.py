import joblib
import pandas as pd
import nltk
import numpy as np
import re
import string
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import nltk
from nltk.corpus import stopwords
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import spacy
from fastai.text.all import *
import requests
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
nlp = spacy.load("en_core_web_sm")


def remove_stopwords(text):
    text = " ".join(word for word in text.split(" ") if word not in stop_words)
    return text


def lemmatiz(text):
    return lemmatizer.lemmatize(text)


# Load the saved model
models = joblib.load("./models.joblib")


# Define a function to preprocess the title and body of a news article
def preprocess(title, body):
    # Count of sentences in title and body
    sentCnt_tit = len(sent_tokenize(title))
    sentCnt = len(sent_tokenize(body))

    # Count of words in title and body
    wrdCnt_tit = len(word_tokenize(title))
    wrdCnt = len(word_tokenize(body))

    # Count of stop words in title and body
    stop_words = set(stopwords.words("english"))
    swCnt_tit = len([w for w in word_tokenize(title) if w in stop_words])
    swCnt = len([w for w in word_tokenize(body) if w in stop_words])

    # Count of named entities in title and body
    entCnt = len(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(body))).leaves())

    # Average length of words in title and body
    avLen_tit = np.mean([len(w) for w in word_tokenize(title)])
    avLen = np.mean([len(w) for w in word_tokenize(body)])

    # Sentiment analysis of title and body
    sid = SentimentIntensityAnalyzer()
    sent_tit = sid.polarity_scores(title)
    sent = sid.polarity_scores(body)
    neg_tit, neu_tit, pos_tit, compound_tit = (
        sent_tit["neg"],
        sent_tit["neu"],
        sent_tit["pos"],
        sent_tit["compound"],
    )
    neg, neu, pos, compound = sent["neg"], sent["neu"], sent["pos"], sent["compound"]

    # POS tag count in title and body
    title_nouns, title_verbs, title_adjectives = get_pos_counts(title)
    text_nouns, text_verbs, text_adjectives = get_pos_counts(body)

    # Create a dictionary of the features
    features = {
        "sentCnt_tit": sentCnt_tit,
        "wrdCnt_tit": wrdCnt_tit,
        "swCnt_tit": swCnt_tit,
        "avLen_tit": avLen_tit,
        "sentCnt": sentCnt,
        "wrdCnt": wrdCnt,
        "swCnt": swCnt,
        "entCnt": entCnt,
        "avLen": avLen,
        "neg": neg,
        "neu": neu,
        "pos": pos,
        "compound": compound,
        "title_nouns": title_nouns,
        "title_verbs": title_verbs,
        "title_adjectives": title_adjectives,
        "text_nouns": text_nouns,
        "text_verbs": text_verbs,
        "text_adjectives": text_adjectives,
    }
    return pd.DataFrame(features, index=[0])


# Define a function to get the POS counts of a given text
def get_pos_counts(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = {"NN": 0, "VB": 0, "JJ": 0}
    for word, tag in pos_tags:
        if tag.startswith("NN"):
            pos_counts["NN"] += 1
        elif tag.startswith("VB"):
            pos_counts["VB"] += 1
        elif tag.startswith("JJ"):
            pos_counts["JJ"] += 1
    return (pos_counts["NN"], pos_counts["VB"], pos_counts["JJ"])


# Define a function to make predictions using the saved model
def predict(title, body):
    # Preprocess the title and body to extract the features
    features = preprocess(title, body)
    # Make predictions using the trained models
    y_pred = np.zeros((len(features), len(models)))
    for i, model in enumerate(models):
        y_pred[:, i] = model.predict(features)
    # Combine the predictions using averaging
    y_pred_avg = np.mean(y_pred, axis=1)
    # Transform the predicted binary output back to 'fake' or 'real'
    y_pred_label = [1 if pred >= 0.5 else 0 for pred in y_pred_avg]
    # Print the predicted label and probability of each label
    proba = {"fake": y_pred_avg[0], "real": 1 - y_pred_avg[0]}
    print("Predicted label:", y_pred_label)
    print("Probability of each label:", proba)
    return y_pred_label, proba


def ensemble_predict(title, body):
    # Load the saved models
    models = joblib.load("./models.joblib")
    spacy = WordTokenizer()
    tkn = Tokenizer(spacy)
    tit_model = load_learner("./titlechecker.pkl")
    text_model = load_learner("./txtchecker(1).pkl")

    # Preprocess the title and body to extract features for the ML models
    features = preprocess(title, body)

    # Make predictions using the trained ML models
    y_pred = np.zeros((len(features), len(models)))
    for i, model in enumerate(models):
        y_pred[:, i] = model.predict(features)

    # Combine the predictions using averaging
    y_pred_avg = np.mean(y_pred, axis=1)
    y_pred_avg = 1 - y_pred_avg
    # Transform the predicted binary output back to 'fake' or 'real'
    ml_prediction = "Fake" if y_pred_avg[0] >= 0.9 else "Real"

    # Use NLP models to make prediction
    final_tit = tkn(lemmatiz(remove_stopwords(title)))
    ppst_tit = ""
    for i in final_tit:
        ppst_tit += i + " "
    pred_tit, tensor_tit, probs_tit = tit_model.predict(ppst_tit)
    final_txt = tkn(lemmatiz(remove_stopwords(body)))
    ppst_txt = ""
    for i in final_txt:
        ppst_txt += i + " "
    pred_txt, tensor_txt, probs_txt = text_model.predict(ppst_txt)

    # Average the predicted probabilities of title and body
    avg_probs = 0.5 * probs_tit + 0.5 * probs_txt

    # Get the index of the class with highest probability
    nlp_pred_idx = torch.argmax(avg_probs).item()
    nlp_pred_idx = 1 - nlp_pred_idx
    # Map the index to the corresponding class label
    nlp_prediction = "Real" if nlp_pred_idx == 1 else "Fake"
    ensemble_prediction = nlp_prediction
    # Combine the predictions using majority voting
    if ml_prediction == nlp_prediction:
        ensemble_prediction = ml_prediction
    else:
        ensemble_prediction = "Fake"  # Choose "Fake" as the tiebreaker

    # Return the ensemble prediction, probabilities, and individual predictions
    resp = {
        "ensemble_prediction": ensemble_prediction,
        "ensemble_probabilities": {
            "Fake": np.mean([y_pred_avg[0], avg_probs[1]]),
            "Real": np.mean([1 - y_pred_avg[0], avg_probs[0]]),
        },
        "ml_prediction": ml_prediction,
        "ml_probabilities": {"Fake": y_pred_avg[0], "Real": 1 - y_pred_avg[0]},
        "nlp_prediction": nlp_prediction,
        "nlp_probabilities": {"Fake": avg_probs[1], "Real": avg_probs[0]},
    }

    return resp


class NewsInput(BaseModel):
    title: str
    body: str


app = FastAPI()
@app.get("/home")
async def home():
    return "Hello World"

@app.post("/predict")
async def predict_news(news: NewsInput):
    title = news.title
    body = news.body
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    query_params = {"query": title, "key": "AIzaSyAilvlYfblaFQlTAesyYX0jZcaJmMjcfgo"}

    response = requests.get(url, params=query_params)
    print(response)
    if response.json():
        return response.json()["claims"][0]["claimReview"][0]
    else:
        return ensemble_predict(title, body)
