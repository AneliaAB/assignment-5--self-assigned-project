#%%
import spacy
import pandas as pd
import os
import pickle

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
def extract_stock():
    #create corpus from data
    data = pd.read_csv(os.path.join("..","in","stock_data.csv"))
    corpus = []

    for ind in data.index:
        text = data['Text'][ind]
        corpus.append(text)

    #%%
    #extract stock name from text
    for text in corpus:
        doc = nlp(text)

        for ent in doc.ents:
            entity = ent.text
            lable = ent.label_
    
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    stock = input("Name of stock or company: ")
    
    for text in corpus:
        if stock in text:
            test_sentence = loaded_vectorizer.transform([text])
            print(loaded_model.predict(test_sentence))

#%%
extract_stock()

# %%
#generates a prediction based on prompt
def classify_sentance():
    entity, lable = extract_stock()

    # load the vectorizer
    loaded_vectorizer = pickle.load(open('../models/vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('../models/classification.model', 'rb'))

    tweet = input("Write a sentance to generate lable with Neural Network classifier: ")
    test_sentence = loaded_vectorizer.transform([sentence])
    print(loaded_model.predict(test_sentence))

# %%
#classify_sentance()
#NKD looking like a good short. Failed to break price level resistance at 116 today yealds a negative sentiment