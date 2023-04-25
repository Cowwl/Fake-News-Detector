# Ensemble News Classifier with NLP and ML models
This project is an ensemble news classifier that predicts whether a news article is real or fake. It combines the predictions of machine learning (ML) models and natural language processing (NLP) models to make the final prediction.

# Dependencies
* joblib
* pandas
* nltk
* numpy
* re
* string
* fastapi
* pydantic
* requests
* spacy
* fastai
# Usage
Clone the repository
Install the dependencies using pip install -r requirements.txt
Run the main.py file using the command uvicorn main:app --reload on the terminal
The app will start running on http://127.0.0.1:8000
## Endpoints
### /home
This endpoint is used to test if the app is running successfully. It returns a "Hello World" message.

### /predict
This is the main endpoint that accepts POST requests with JSON data containing the title and body of the news article. It returns a JSON response with the predicted label (Real or Fake) and the probabilities of each label. If the Google Fact-Check API returns a result for the given title, the response also includes the fact-check result.

#How it works
The app preprocesses the title and body of the news article to extract various features such as the count of sentences, words, stop words, named entities, and sentiment analysis. These features are used as inputs for the ML models. The ML models are trained on a dataset of real and fake news articles and predict the probability of the article being fake.

The NLP models use tokenization, lemmatization, and stop word removal to preprocess the title and body. The preprocessed text is then passed through pre-trained models to predict the probability of the article being real or fake.

The results of the ML and NLP models are combined using averaging and majority voting to make the final prediction.

If the Google Fact-Check API returns a result for the given title, the app returns the fact-check result instead of making a prediction using the ML and NLP models.

# Conclusion
This app provides an accurate and reliable way to predict the authenticity of news articles using both ML and NLP models. It can be used to combat fake news and misinformation, which is a growing problem in the digital age.
