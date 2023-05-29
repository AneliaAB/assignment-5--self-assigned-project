#%%
#handling text
import spacy
from spacy import displacy
#data handling and visualization
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#saving and loading models
import pickle

#%%
#%%
df = pd.read_csv(os.path.join("..", "data", "stocks.tsv"), sep="\t")

symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()

stops = ["two"]
nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
patterns = []
for symbol in symbols:
    patterns.append({"label": "STOCK", "pattern": symbol})
    for l in letters:
        patterns.append({"label": "STOCK", "pattern": symbol+f".{l}"})
for company in companies:
    if company not in stops:
        patterns.append({"label": "COMPANY", "pattern": company})

ruler.add_patterns(patterns)

#%%
def create_corpus():
    data = pd.read_csv(os.path.join("..","in","stock_data.csv"))
    corpus = []

    for ind in data.index:
        text = data['Text'][ind]
        corpus.append(text)   
    
    return corpus

#%%
def extract_stock():
    #load vectorizer and model
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    corpus = create_corpus()

    sentiments = []
    stock = input("Name: ")
    
    stock_texts = []
    for text in corpus:
        doc = nlp(text)

        for ent in doc.ents:
            entity = ent.text
            lable = ent.label_
        
            if entity == stock in text:
                stock_texts.append(text)
                test_sentence = loaded_vectorizer.transform([text])
        
                sentiment = loaded_model.predict(test_sentence)
                sentiments.append(sentiment)
    
    combined_text = " ".join(stock_texts)
    doc = nlp(combined_text)
    displacy.serve(doc, style="ent", auto_select_port=True)
    joined_array = np.concatenate(sentiments)
    return joined_array

# %%
def create_df():
    sentiments = extract_stock().tolist()
    labels = ["positive", "negative"]

    values = [sentiments.count(1), sentiments.count(-1)]
    plt.bar(labels, values)

    plt.title('Distribution of positive and negative sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Show the plot
    plt.savefig('../out/sentiment_distribution.png')

create_df()