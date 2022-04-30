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
#trainCSV.isnull().sum()
#trainCSV.shape


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

#print(testCSV.head())
#print(trainCSV.head())


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

########### vectorization process

# instantiates the vectorizer,
count_vectorizer = CountVectorizer(
    min_df=3, # Takes out words that only appear 3 or less times
    max_df=0.5 # Takes out words that are in more than 50%, since they add no valuable data

)
# Fits the vectorized with train data
train_vectors = count_vectorizer.fit_transform(trainCSV['text'])

# Gets a list of all the words in the vector
vector_features = count_vectorizer.get_feature_names()
print("Vector features: ",vector_features) # Prints all the words fit into the in the vectorizer
print("Feature Counts: ",len(vector_features)) # Prints the amount of words in the vectorizer

# Converts the vectorized data matrix to array
train_vec_arr = train_vectors.toarray()
# Puts the vectorized data into the dataframe
train_vec_dataframe = pd.DataFrame(data=train_vec_arr,columns = vector_features)
# Combines vector dataframe to train dataframe
trainCSV = pd.concat([trainCSV, train_vec_dataframe], axis=1, join='inner')

# Exports and prints for viewing
trainCSV.to_csv("./data/train_vectorized.csv")
print("Vectorized Train Data:\n\n" ,trainCSV.head())

# Vectorizes the test data
test_vectors = count_vectorizer.transform(testCSV["text"])
# Converts the vectorized data matrix to array
test_vec_arr = test_vectors.toarray()
# Puts the vectorized data into the dataframe
test_vec_dataframe = pd.DataFrame(data=test_vec_arr,columns = vector_features)
# Combines vector dataframe to train dataframe
testCSV = pd.concat([testCSV, test_vec_dataframe], axis=1, join='inner')





