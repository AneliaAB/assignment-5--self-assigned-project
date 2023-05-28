#%%
import spacy
import pandas as pd

#%%
df = pd.read_csv("data/stocks.tsv", sep="\t")

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

#%%

doc = nlp(text)
for ent in doc.ents:
    print (ent.text, ent.label_)

#%%
