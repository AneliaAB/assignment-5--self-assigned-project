#%%
from zipfile import ZipFile
# # system tools
import os
import sys
sys.path.append("..")

# data munging tools
import pandas as pd
#import utils.classifier_utils as clf
import numpy as np

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import learning_curve

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
    data = pd.read_csv(os.path.join("..","in","stock_data.csv"))
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
    classifier = MLPClassifier(activation = "relu",
                               solver="sgd",
                           hidden_layer_sizes = (15,),
                           max_iter=1000,
                           random_state = 42)

    classifier = classifier.fit(X_train_feats, y_train)

    pickle.dump(classifier, open('../models/classification.model', 'wb'))
    pickle.dump(vectorizer, open('../models/vectorizer.pickle', 'wb'))

    # Generate the learning curve
    train_sizes, train_scores, val_scores = learning_curve(classifier, X_train_feats, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate the mean and standard deviation of the training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.savefig("../out/loss_accuracy_curve_sgd.png", format="png") # specify filetype explicitly
    plt.show()

    plt.close()
    

#%%
train_classifier()

#%%
#generates classification report and saves it in text format
def classification_report():
    #load train and test data
    X_train, X_test, y_train, y_test = prepare_data()

    #load vectorizer and model
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    X_train_feats = loaded_vectorizer.fit_transform(X_train)
    X_test_feats = loaded_vectorizer.transform(X_test)

    y_pred = loaded_model.predict(X_test_feats)
    classifier_metrics = metrics.classification_report(y_test, y_pred)

    #save classification report
    text_file = open("../out/classification_report_relu.txt", "w")
    n = text_file.write(classifier_metrics)
    text_file.close()
# %%
classification_report()

# %%
#generates a prediction based on prompt
def classify_sentance():
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    sentence = input("Write a sentance to generate lable with Neural Network classifier: ")
    test_sentence = loaded_vectorizer.transform([sentence])
    print(loaded_model.predict(test_sentence))

# %%
#classify_sentance()
#NKD looking like a good short. Failed to break price level resistance at 116 today yealds a negative sentiment