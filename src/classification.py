from zipfile import ZipFile
# # system tools
import os
import sys
sys.path.append("..")

# data munging tools
import pandas as pd
#import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Visualisation
import matplotlib.pyplot as plt

#saving reports
import pickle 

# %%
#unzipping data folder
#zip = ZipFile('../data/stock_market.zip')
#zip.extractall("../in/") 

#%%
#loads the data and splits it into train and test
def prepare_data():
    #load data
    data = pd.read_csv(os.path.join("..","in","stock_data.csv"), index_col=0)
    X = data["Text"]
    y = data["Sentiment"] 

    #split into test and train 
    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                    y,          
                                                    test_size=0.2,   
                                                    random_state=42)
    
    return X_train, X_test, y_train, y_test 

#%%
#trains classifier and saves model/vectorizer
def train_classifier():
    #load test and train data
    X_train, X_test, y_train, y_test = prepare_data()

    vectorizer = TfidfVectorizer(ngram_range = (1,2),     
                            lowercase =  True,       
                            max_df = 0.95,          
                            min_df = 0.05,           
                            max_features = 100) 

    #fit the vectorizer
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    #Initiate report and prediction 
    classifier = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (20,),
                           max_iter=1000,
                           random_state = 42)

    classifier.fit(X_train_feats, y_train)

    pickle.dump(classifier, open('../models/neural_network/classification.model', 'wb'))
    pickle.dump(vectorizer, open('../models/neural_network/vectorizer.pickle', 'wb'))

#%%
train_classifier()

#%%
#generates classification report and saves it in text format
def classification_report():
    #load train and test data
    X_train, X_test, y_train, y_test = prepare_data()

    #load vectorizer and model
    loaded_vectorizer = pickle.load(open('../models/neural_network/vectorizer.pickle', 'rb'))
    loaded_model = pickle.load(open('../models/neural_network/classification.model', 'rb'))

    X_train_feats = loaded_vectorizer.fit_transform(X_train)
    X_test_feats = loaded_vectorizer.transform(X_test)
    
    feature_names = loaded_vectorizer.get_feature_names_out()

    y_pred = loaded_model.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)

    #save classification report
    text_file = open("../out/neural_network/classification_report.txt", "w")
    n = text_file.write(classifier_metrics)
    text_file.close()
# %%
classification_report()
