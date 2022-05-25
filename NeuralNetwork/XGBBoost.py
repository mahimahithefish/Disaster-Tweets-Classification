# importing files
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
import pandas as pd
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LeakyReLU
from xgboost import XGBRegressor


trainCSV = pd.read_csv("../data/train_model_ready.csv")

y_train_full = trainCSV.target
X_train_full = trainCSV.drop(["target", "clean_tweet", "Unnamed: 0"], axis=1)

"""
# Removes the next word cause it causes bugs with the nerual network
X_train_full.drop(["next"], axis=1)
"""

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=1, train_size=0.8, test_size=0.2)


XGBmodeltrain = XGBRegressor(
    n_estimators=10000,
    learning_rate= 0.01,
    #eval_metric="error",
    objective="binary:logistic",
    #disable_default_eval_metric=True
)

# Fit the model
XGBmodeltrain.fit(X_train,y_train, early_stopping_rounds=25, eval_set=[(X_valid,y_valid)])

valPredictions = XGBmodeltrain.predict(X_valid)


for i in range(len(valPredictions)) :
    valPredictions[i] = round(valPredictions[i])

print(valPredictions)

meanError = accuracy_score(y_valid, valPredictions)

print("Accuracy: ", meanError)

# Accuracy 79.18581746552856%


testCSV = pd.read_csv("../data/test_model_ready.csv")

testData = testCSV.drop(["clean_tweet"])
