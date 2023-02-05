import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


import pandas as pd
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from disaster_response", engine)
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    return X, Y, Y.columns

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('text_pipeline', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
        ])
    
    parameters = {
    #'text_pipeline__ngram_range':((1,1),(1,2)),
    'clf__estimator__n_estimators': [5, 10],
    #'clf__estimator__min_samples_split': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    pred = model.predict(X_test)
    for j, col in enumerate(category_names):
        print('='*25,col,'='*25)
        print(classification_report(Y_test[col], pred[:,j]))


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)
    print('modelo guardado!')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()