import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
from IPython import display


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    In this function we are loading both categories data set and messages data set and merge two data sets
    INPUT:
        - Messages.csv, categories.csv
    OUTPUT:
        - df
    """
    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how='left', on='id')
    return df


def clean_data(df):
    """
    We are cleaning our data in this function. We will create labels for categories and drop any duplicates. Then we will concatenate category labels with values to messages table
    INPUT: 
        - df
    OUTPUT: 
        - Cleaned df
    """

    # create a dataframe of the 36 individual category columns and split it at';' level
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # apply a lambda function that takes everything up to the second 
    # to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # clip the value 2 in related column to 1
    categories = categories.clip(0,1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    In this function we are saving the data to the SQL database
    
    INPUT:
        - disaster messages
    OUTPUT:
        - SQLite database
    """
    engine = create_engine('sqlite:///'+database_filename)

    conn = engine.connect()

    # Save dataframe to sql table
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

    conn.close()
    engine.dispose()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
    



if __name__ == '__main__':
    main()
