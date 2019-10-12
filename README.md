# Disaster-Response-Pipeline-prediction

### Table of Contents

1. [Libraries used for the project](#libraries)
2. [Objective](#motivation)
3. [File Descriptions](#files)
4. [Summary Of Models](#Models)
5. [Instructions](#Instructions)
6. [Acknowledgements](#acknowledgements)

## Libraries used for the project <a name="libraries"></a>

Following python libraries:

1. Collections
2. Matplotlib 
3. NLTK
4. NumPy
5. Pandas
6. Seaborn
7. Sklearn
8. Pipeline, train_test_split, GridSearchCV, LinearRegression, r2_score, classification_Report, accuracy_score, recall_score, precision_score, f1_score, TfidfVectorizer, MultiOutputClassifer, AdaBoostClassifier, GradientBoostingClassifier, BaseEstimatore, TransformerMixin, MLPClassifier
9. SQLAlchemy


I used the Anconda python distribution with python 3.0

## Objective<a name="motivation"></a>

The objective of this project is to build a model and classify messages during a disaster. We have been given disaster twitter messages data set which have 36 pre-defined categories. With the help of the model, we can classify the message to these categories and send the message to the appropriate disaster relief agency. For example, we do not want Medical Help message to food agency as they wont be able to help the person in time. 

This project will involve building an ETL pipeline and Machine Learning pipeline. Objective of this task is also multiclassification. We want one message to be classified to multiple categories if needed. 

This data set is  provided to us by [Figure Eight]((https://www.figure-eight.com/)

## File Descriptions <a name="files"></a>




data:
- disaster_message.csv
- disaster_Categories.csv
- DisasterResponse.db
- ETL pipeline Preparation.ipynb
- process_data.py

models:
- train_classifer.py
- ML Pipeline Preparation.ipynb
- glove_vectorizer.py

app:
- templates
  -go.html
  -master.html
-run.py

GLoVe:
- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt


## Summary Of Models<a name="Models"></a>

We are using two models to classify messages

1. In the first model, we use Tfidf vectorizer to transform messages and then we use Adaboost Classifier to classify messages

2. In the second model, we use pretrained GLoVe embedding in MLP neural network to transform messages and then we classify them .


## Instructions<a name = "Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

  
## Acknowledgements<a name="acknowledgements"></a>

- This data set is  provided to us by [Figure Eight]((https://www.figure-eight.com/)
- test train split error https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam
- Pandas string split https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html
