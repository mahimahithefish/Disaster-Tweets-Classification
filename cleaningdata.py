## All purpose libraries
import pandas as pd

## NLP Libraries
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# importing files
from sklearn.feature_extraction.text import CountVectorizer

testCSV = pd.read_csv("./data/testOG.csv")
trainCSV = pd.read_csv("./data/trainOG.csv")

# ########## Parses Keyword data
def keyWordData(df):
    # -1 means there is no keyword
    # 0 means there's a keyword, but it's not in the tweet
    # 1 means there's a keyword, and it's in the tweet
    hasKeyWordArr = []

    for index, row in df.iterrows():
        tweet = row.text.lower()
        keyword = row.keyword
        if not keyword or isinstance(keyword, float) or keyword == "":
            hasKeyWordArr.append(-1)
        elif keyword.lower() in tweet:
            hasKeyWordArr.append(1)
        else:
            hasKeyWordArr.append(0)

    df["HasKeyWord"] = hasKeyWordArr
    df.drop("keyword", axis=1, inplace=True)


keyWordData(trainCSV)
keyWordData(testCSV)

# ########## Parses Location Data
def locationData(df):
    # 0 means there is no location
    # 1 means there there's a location
    hasLocationArr = []

    for index, row in df.iterrows():
        location = row.location
        if not location or isinstance(location, float) or location == "":
            hasLocationArr.append(0)
        else:
            hasLocationArr.append(1)

    df["HasLocation"] = hasLocationArr
    df.drop("location", axis=1, inplace=True)
locationData(trainCSV)
locationData(testCSV)

# ########## Clean data
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
    # Simplifying the tweets by stemming
    tweets = df['text'].tolist()
    stemmer = PorterStemmer()
    stemmedtweets = []
    for tweet in tweets:
        stemmedtweets.append(stemmer.stem(tweet))
    stemmedtweets = pd.DataFrame(stemmedtweets)
    df['clean_tweet'] = stemmedtweets # appending a new row to the given dataframe


cleanText(testCSV)
cleanText(trainCSV)

# Exporting for viewing the data - this way we can see the original and the cleaned tweets side by side
trainCSV.to_csv("./data/train.csv")
testCSV.to_csv("./data/test.csv")

train = trainCSV['clean_tweet'].tolist()
traintarget = trainCSV['target'].tolist()

from sklearn.feature_extraction.text import TfidfVectorizer  # Coverts text into TF-ID matrices
TfidVectorizer = TfidfVectorizer(min_df=3)

# Fits the vectorized with train data
train_vectors = TfidVectorizer.fit_transform(trainCSV['text'])

# Gets a list of all the words in the vector
vector_features = TfidVectorizer.get_feature_names()
print("Vector features: ", vector_features)  # Prints all the words fit into the in the vectorizer
print("Feature Counts: ", len(vector_features), "\n\n")  # Prints the amount of words in the vectorizer

# Converts the vectorized data matrix to array
train_vec_arr = train_vectors.toarray()
# Puts the vectorized data into the dataframe
train_vec_dataframe = pd.DataFrame(data=train_vec_arr, columns=vector_features)
# Combines vector dataframe to train dataframe
trainCSV = pd.concat([trainCSV, train_vec_dataframe], axis=1, join='inner')



# Vectorizes the test data
test_vectors = TfidVectorizer.transform(testCSV["text"])
# Converts the vectorized data matrix to array
test_vec_arr = test_vectors.toarray()
# Puts the vectorized data into the dataframe
test_vec_dataframe = pd.DataFrame(data=test_vec_arr, columns=vector_features)
# Combines vector dataframe to train dataframe
testCSV = pd.concat([testCSV, test_vec_dataframe], axis=1, join='inner')

# ##### Drops the text and id column as it isn't needed for the model
testCSV.drop("id", axis=1, inplace=True)
trainCSV.drop("id", axis=1, inplace=True)
testCSV.drop("text", axis=1, inplace=True)
trainCSV.drop("text", axis=1, inplace=True)


# Exports and prints for viewing
print("Vectorized Train Data:\n\n", trainCSV.head())
trainCSV.to_csv("./data/train_model_ready.csv")
testCSV.to_csv("./data/test_model_ready.csv")

