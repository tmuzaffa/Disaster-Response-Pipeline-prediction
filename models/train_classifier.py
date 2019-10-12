import sys
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
from IPython import display




from sqlalchemy import create_engine
from collections import Counter


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    This function loads the data from SQLite database and divide the data into X, Y, and category names
    
    INPUTS:
        SQLite database file path
    RETURNS:
        X
        Y
        category_names 
    """
    # load data from database

    engine = create_engine('sqlite:///'+database_filepath)

    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    
    # Get names of all categories
    cat_names = Y.columns.tolist()
    return X, Y, cat_names


def tokenize(text):
    """
   This function will clean and tokenize message for modeling. In this model we are replaing all punctuations with a blank space and then are tokenizing and lemmatizing using parts of speech verb
   . After that, we stem twords using PorterStemmer
    
    INPUTS:
        messages text data
    RETURNS:
        clean_tokens
    """
    # for this function we will be using the same cleaning steps as used in clean_tokenzie exercise during the NLP pipeline lesson
    #as learnt in building NLP pipeline process, we will clean data first by first replacing anything that is not a digit or an alphabet with space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Now that the text has been cleaned i.e. punctuations have been removed we can tokenize
    tokens = word_tokenize(text)
    #after using tokenization, we can initiate lemmatizer
    lemmatizer =  WordNetLemmatizer()
    #Now we can initiate Stemming
    stemmer = PorterStemmer()
    clean_tokens = []
    for tok in tokens:
        #we will now lemmatize token using verb, and make it lower case and remove empty space
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        #now we can stem token by using stemmer
        clean_tok = stemmer.stem(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    In this function we are building our model. We have left the capability of adding more models to this functions later, if user so desire. 
    
    INPUT: 
        - we are using model as our input, this allows user to add more models to this function and then select them using the model parameter and see the result
    OUPUT:
        - Gridsearch model
    """
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    parameters = {'tfidf__ngram_range': ((1, 1), (1,3))}
    

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return cv




def evaluate_model(model, X_test, Y_test, cat_names):
    """
    In this function we are evaluating the performance of our model/models built in the above function
    INPUT: 
        - Test files for X and Y, categorynames and model
    OUTPUT:
        - f1 score, precision, and recall
    """
    Y_predict = model.predict(X_test)
    cat_names = Y_test.columns.tolist()
    df_y_predict = pd.DataFrame(Y_predict, columns = cat_names)
    df_y_predict.head()
    for i in range(len(cat_names)):
        print(cat_names[i],\
          '\n',\
          classification_report(Y_test.iloc[:,i], df_y_predict.iloc[:,i]))

def save_model(model, model_filepath):
    """
    In this function we are saving the optimized model to the specified path
    INPUT: 
        - model, and model path
    OUTPUT:
        - Path
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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