## All purpose libraries
import pandas as pd
import seaborn as sns

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


import matplotlib.pyplot as plt
labels= ['Disaster tweets', 'Non-disaster tweet']

colors=['blue', 'yellow']

sizes= [len(dis_tweet), len(non_dis)]

plt.pie(sizes,labels=labels, colors=colors, startangle=90, shadow=True,explode=(0.1, 0.1), autopct='%1.2f%%')

plt.title('Tweet Target value distribution')

plt.axis('equal')

plt.show()


