#%%
import spacy
import pandas as pd
import os
import pickle
from spacy import displacy

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
    stock = input("Name: ")
    lable = input("Type: ")
    
    
    for text in corpus:
        doc = nlp(text)

        for ent in doc.ents:
            entity = ent.text
            lable = ent.label_
        
            if entity == stock in text:
                if lable == lable.upper():
                    displacy.render(doc, style="ent")
                    test_sentence = loaded_vectorizer.transform([text])
            
                    print(loaded_model.predict(test_sentence))

#%%
extract_stock()

# %%
def create_df(path_to_folder):
    filenames = os.listdir(path_to_folder)

    categories = []
    for filename in filenames:
        if path_to_folder == "../data/bleached_corals":
            categories.append(0)
        else:
            categories.append(1)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    
    return df

healthy_df = create_df("../data/healthy_corals")
bleached_df = create_df("../data/bleached_corals")

df = pd.concat([bleached_df, healthy_df])
df_plt = df["category"].replace({0: 'bleached', 1: 'healthy'}).value_counts().plot.bar() 
df_plt.figure.savefig("../out/visualizing_dataframe.png", dpi=300, bbox_inches='tight') # specify filetype explicitly