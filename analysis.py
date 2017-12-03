from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
import requests
import nltk
import pandas as pd
import re
import os

STEMMER = PorterStemmer()

def search(query):
    return requests.get('http://api.nytimes.com/svc/search/v2/articlesearch.json', params={'q': query, 'api-key': os.environ['NYTKEY']}).json()

def tokenize(text):
    tokens = nltk.word_tokenize(re.sub('[^a-zA-Z]', ' ', text))
    return [STEMMER.stem(t) for t in tokens]

def analyze(path):

    # load in a dataset into both testing and training data frames
    test_data_frame = pd.read_csv(path, header=None, delimiter='\t', quoting=3)
    test_data_frame.columns = ['Text']
    train_data_frame = pd.read_csv('training.csv', header=None, delimiter='\t', quoting=3)
    train_data_frame.columns = ['Sentiment', 'Text']

    # Use tokenize function on training dataset
    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tokenize,
        lowercase=True,
        stop_words='english',
        max_features=85
    )

    # Combine all the data we need to look at, training and to be predicted, into one collection
    fit_data = vectorizer.fit_transform(train_data_frame.Text.tolist() + test_data_frame.Text.tolist())
    fit_data = fit_data.toarray()

    # Predict the sentiment of unlabeled data
    model = LogisticRegression()
    model = model.fit(X=fit_data[0:len(train_data_frame)], y=train_data_frame.Sentiment)
    pred = model.predict(fit_data[len(train_data_frame):])

    # read each row of the text column along with the predicted sentiment
    # 1 => positive
    # 0 => negative
    for text, sentiment in zip(test_data_frame.Text, pred):
        print(sentiment, text)
