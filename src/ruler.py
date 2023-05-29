#%%
import spacy
import pandas as pd
import os
import pickle
from spacy import displacy
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv(os.path.join("..", "data", "stocks.tsv"), sep="\t")

symbols = df.Symbol.tolist()
companies = df.CompanyName.tolist()
print (symbols[:10])

#%%
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
def gather_input():
    stock = input("Name: ")
    label = input("Type: ")

    return stock, label 

#%%
def extract_stock():
    #load vectorizer and model
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    #create corpus
    data = pd.read_csv(os.path.join("..","in","stock_data.csv"))
    corpus = []

    for ind in data.index:
        text = data['Text'][ind]
        corpus.append(text)
    
    #generate stock/company name
    stock, label = gather_input()

    sentiments = []
    
    for text in corpus:
        doc = nlp(text)

        for ent in doc.ents:
            entity = ent.text
            lable = ent.label_
        
            if entity == stock in text:
                if lable == lable.upper():
                    displacy.render(doc, style="ent")
                    test_sentence = loaded_vectorizer.transform([text])
            
                    sentiment = loaded_model.predict(test_sentence)
                    sentiments.append(sentiment)
    joined_array = np.concatenate(sentiments)
    return joined_array
#%%
extract_stock()

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
    plt.savefig('plot.png')

create_df()
# %%
