import pandas as pd

# importing files
testCSV = pd.read_csv("/Users/tasnimmahi/Desktop/Disaster-Tweets-Classification/data/test.csv")
trainCSV = pd.read_csv("/Users/tasnimmahi/Desktop/Disaster-Tweets-Classification/data/train.csv")

train = trainCSV['clean_tweet'].tolist() # Cleaned data are the inputs
traintarget = trainCSV['target'].tolist() # the tweet classification is considered as the output

from sklearn.feature_extraction.text import TfidfVectorizer  # Coverts text into TF-ID matrices
vectorizer = TfidfVectorizer(min_df=3)

x_train = vectorizer.fit_transform(train)
x_test = vectorizer.transform(testCSV.clean_tweet)

from sklearn.linear_model import SGDClassifier  # Using SGD model to train the data - Logistic regression

model = SGDClassifier(loss="log", max_iter=1000, alpha=0.0001, random_state=42)
model_train = model.fit(x_train, traintarget)  # training the model
pred_test = model_train.predict(x_test) # predicting the values

ids = testCSV["id"]
submission_df = pd.DataFrame({"id": ids, "target": pred_test})
submission_df.reset_index(drop=True, inplace=True)

submission_df.to_csv("submission.csv", index=False)  # Exporting file with the newly predicted values
