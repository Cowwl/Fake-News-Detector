from fastapi import FastAPI
from pydantic import BaseModel
from fastai.text.all import *
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import requests
nltk.download('wordnet')
nltk.download('wordnet2022')
# ! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
nlp = load('en_core_web_sm')

app = FastAPI()

class Query(BaseModel):
    inp: str

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

def lemmatiz(text):
    return lemmatizer.lemmatize(text)

def predict(query):
    spacy = WordTokenizer()
    tkn = Tokenizer(spacy)
    if len(query) < 100:
        model = load_learner('titlechecker.pkl')
    else:
        model = load_learner('export.pkl')
    final = tkn(lemmatiz(remove_stopwords(query)))
    ppst=""
    for i in final:
        ppst+=i+' '
    prediction, tensor, probabilities = model.predict(ppst)
    return json.dumps({
        "prediction": "Real" if prediction == 0 else "Fake",
        "tensor": tensor.tolist(),
        "probabilities": probabilities.tolist()
    })

@app.post("/prediction")
async def get_prediction(query: Query):
    response = requests.get(url, params={"query": query.inp, "key": "YOUR_API_KEY"})
    if response.json():
        return response.json()['claims'][0]['claimReview'][0]
    else:
        return predict(query.inp)