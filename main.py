## All purpose libraries
import pandas as pd

## NLP Libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# importing files
from sklearn.feature_extraction.text import CountVectorizer

testCSV = pd.read_csv("./data/test.csv")
trainCSV = pd.read_csv("./data/train.csv")

# viewing data
trainCSV.isnull().sum()
trainCSV.shape


# Clean data
def cleanText(df):
    stopWords = set(stopwords.words("english"))

    for index, row in df.iterrows():
        curText = row.text

        # Makes lowercase
        curText = curText.lower()

        # removing stop words
        curText = " ".join(filter(lambda x: x not in stopWords, curText.split()))

        # removing all words starting with @
        curText = " ".join(filter(lambda x: x[0] != "@", curText.split()))

        # removes all links
        webStart = "http"
        curText = " ".join(filter(lambda x: x[0:len(webStart)] != webStart, curText.split()))

        # removing all non alpha numeric char exlcuding period
        curText = re.sub(r'[^a-z0-9. ]+', '', curText)
        # removing "..." (multiple periods in a row)
        curText = re.sub(r'([.])\1+', '', curText)
        # removing multiple spaces in a row
        curText = re.sub(r'([ ])\1+', '', curText)

        df.at[index, 'text'] = curText


cleanText(testCSV)
cleanText(trainCSV)

print(testCSV.head())
print(trainCSV.head())


def stemSentence(tweets):
    stemmer = PorterStemmer()
    edited_tweets = []
    for tweet in tweets:
        edited_tweets.append(stemmer.stem(tweet))
    return edited_tweets


testlist = testCSV["text"].tolist()
trainlist = trainCSV["text"].tolist()
# stemming the tweets in the list of tweets
testlist = stemSentence(testlist)
trainlist = stemSentence(trainlist)

# vectorization process
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(trainCSV['text'])
test_vectors = count_vectorizer.transform(testCSV["text"])

# print(train_vectors)
# print(test_vectors)



