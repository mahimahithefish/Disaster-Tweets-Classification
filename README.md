# Disaster-Tweets-Classification

In this project, we applied machine learning models to predict Tweets that are about real disasters. The dataset is provided by Kaggle. There are about  10,000 tweets provided. This project is from kaggle.

## Data Cleaning 
In the data cleaning stage, we have coverted all the tweets into lower case letters. We have retained the occurance of internet slang terms and the period punctuation of the tweets. However, using regual expressions, we removed mentions of other users (words that start with "@"), links, alphanumeric characters (excluding the occurances of periods), and removing occurances of multiple periods or spaces in a row. Furthermore, stopwords were also removed using the [nltk](https://www.nltk.org/) (Natural Language Toolkit). 

### Stemming 
All the tweets in our dataset undergone stemming. Stemming is used to simplify each word in a sentece to its common base form without taking account of the context of the words with in the sentence. We used PorterStemmer from the nltk library to complete this. Porterstemer uses the Porter's algorithm which has list of 5 rules, that are applied sequentially to reduce the words to its base form. 

// discuss location parsing and haskeyword function and vecorization maybe???

## Data Modeling 

### Logistic Regression 
We first converted the tweets in the training csv data into TF-IDF matrices. TF-IDF stands for term frequency-inverse document frequency and it is a measure that can quantify the importance of string representations in a document. This was done by using the sklearn library. 

we used SGD (Stochastic Gradient Descent) classifier from the Python sklearn library to train the given tweets. SGD has been applied to large-scale problems encountered in text classification and natural language processing.  In this project, SGD has to be fitted with two arrays as input and output: an array X holding the training tweet's TD- IDF vectors, and an array y holding the target values. After being fitted, the model can then be used to predict new values.

### Neural Networks



## Resources
-  Kaggle: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
-  [SGD logistic regression](https://scikit-learn.org/stable/modules/sgd.html)

