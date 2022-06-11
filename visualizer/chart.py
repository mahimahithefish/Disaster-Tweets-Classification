## All purpose libraries
import pandas as pd
import seaborn as sns
import csv

train = pd.read_csv("/Users/tasnimmahi/Desktop/Disaster-Tweets-Classification/data/train.csv")

tweets = train['clean_tweet'].tolist()
target = train['target'].tolist()

# 1 = is a disaster tweet
# 0 is not a disaster tweet

dis_tweet = [] # Contains tweets that are disaster tweets
non_dis = [] # Contains tweets that are not disaster tweets


for i in range(len(target)):  # Dividing the cleaned tweets into two groups: disaster and non disaster tweets based on
    # the given targets
    if target[i] == 1:
        dis_tweet.append(tweets[i])
    elif target[i] == 0:
        non_dis.append(tweets[i])


from matplotlib import pyplot as plt

# Creating dataset
tweet_type = ['DISASTER TWEETS', 'NON-DISASTER TWEETS']

data = [len(dis_tweet), len(non_dis)]

# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=tweet_type)

# show plot
plt.show()

