# assignment-5--self-assigned-project

## DESCRIPTION
This project trains a classifier on a stock-news dataset gathered from multiple twitter handles regarding economic news. The data is labled with two sentiments: negative (-1) and positive (1) (Yash, 2020). The classifier is then used to generate sentiment analysis - a bar plot visualizing the distribution of positive and negative sentiments. This project allows the user to search through the data and find tweets about a certain stock or company. This can be useful in market analysis or tracking sentiment among users. The project can be applied to data from other platforms as well, as long as the data follows the same structure. It is important to note that the stock-market data (ibid.) is custom labelled, which leaves room for human error or bias. 

**Data** 
The stock-market data (Yash, 2020) can be found on Kaggle via this link: https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset <br >
The stocks data (Mattingly, 2023) can be found in this GitHub repository in the folder ‘data’ under the name of ‘stocks.tsv’: https://github.com/wjbmattingly/freecodecamp_spacy 

## METHODS
This project consists of two scripts. One which trains a classifier on the stock-market data, and one that generates sentiments and a barplot analysis. 

**Training classifier - classification.py**
This script is inspired by previous assignments from this class (particularly assignment 2) and the in-class notebooks. The data is loaded into the script and split into train and test using the ```test_train_split()``` function from *scikit-learn*. The ```MLPClassifier``` in then trained on this data and the model is saved in the ```model``` folder using ```pickle```. A classification report is also saved in the ```out``` folder. <br >

**Sentiments - sentiments.py**
The script begins by loading the ```stocks.tsv``` file and creating a list of patters that will be used by an entity ruler to recognize entities (stocks and companies). This will be used when displaying the text data with, in order to show where in the text a given company or stock is mentioned. 
The stock market data (stock_market.csv) is loaded into the script and the user is asked to provide a name of a stock/company which they wish to do sentiment analysis on. A corpus is created comprised of tweets that mention the company/stock name that the user has provided. The classifier is applied on each tweet and a sentiment is given. The sentiment scores are counted using ```count()``` and a barplot is created in the ```out``` folder. The script also uses the ```displacy.serve()``` function provided by the ```spaCy``` library which displays a visualization of the classified entities in the corpus. You can view the visualization by opening the URL provided when running the script.

This script is inspired by Mattingly, 2023. The Jupyter Notebook can be found via this link: https://github.com/wjbmattingly/freecodecamp_ under the name ```03_01_stock_analysis.ipynb```.

## HOW TO INSTALL AND RUN THE PROJECT
**Installation:**
1. First you need to clone this repository. 
- The datasets are uploaded to the repository, so you won’t need to load them separately
2. Navigate from the root of your directory to ```assignment-5--self-assigned-project```
3. Run the setup file, which will install all the requirements by writing ```bash setup.sh``` in the terminal

**Run the script:** <br >
    4. Navigate to the folder ```src``` by writing ```cd src``` in the terminal, assuming your current directory is ```assignment-5--self-assigned-project``` <br >
    5. First run the script that trains the classifier by writing ```python classification.py``` in the terminal <br >
    6. Then apply the classifier to the data by writing ```python sentiments.py``` in the terminal
a. The user will be asked to provide a name of a stock or company that they wish to do sentiment analysis on. A possible answer is ‘Apple’ (company) or ‘APP’ (stock). You can already find examples in the out folder for Netflix and APP. <br >
b. Open the URL provided by the browser, then press Ctrl+C.

**IMPORTANT:** ```displacy.serve()``` starts a local web server that keeps running until you terminate it manually. To stop the server and regain control of the terminal, you can press ```Ctrl+C``` in most terminals. 

## DISCUSSION OF RESULTS
The results can be found in the ```out``` and ```model``` folder. The ```model``` folder contains the trained classification model and vectorizer. A classification report for the model can be found in the ```out``` folder as well as barplot displaying the distribution of sentiment.

**Classification report**
The performance of the classification model is relatively poor. It shows that the model is better at identifying positive sentiment with f1-score of 0.77, compared to negative sentiment, which only has an f1-score of 0.50. I tried to optimize the performance by changing the number of hidden layers and max iterations, but nothing would get the negative sentiment over 0.50 f1-score. I could get the model to perform even better on the positive sentiment, though that would lower the negative sentiment even more. Therefore, I chose to keep it as it is now. A bigger dataset might have yielded better results.

**Distribution of sentiment**
When the user generates a prompt word - name of a company or stock, which is also present in the ```stocks.tsv``` file - a bar plot is generated showing the distribution of sentiments. Only the comments that mention the prompt word will be included in the final result. The x-axis represents the sentiments, and the y-axis shows the number of tweets. This way the user can quickly get an overview of the general sentiment regarding said stock or company. The flaw in this method is that *all* tweets mentioning the prompt-word would be included, regardless if the sentiment is directed towards that company/stock.

## Referances
Stack Overflow. (n.d.). NotFittedError: TfidfVectorizer - Vocabulary wasn’t fitted. [online] Available at: https://stackoverflow.com/questions/44193154/notfittederror-tfidfvectorizer-vocabulary-wasnt-fitted [Accessed 27 May 2023].

Yash Chaudhary. (2020). <i>Stock-Market Sentiment Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1217821

Mattingly, W. (2023). wjbmattingly/freecodecamp_spacy. [online] GitHub. Available at: https://github.com/wjbmattingly/freecodecamp_spacy [Accessed 26 May 2023].

