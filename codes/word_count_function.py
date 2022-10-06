# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:30:33 2022

@author: tk20
"""

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
sent = 'This is an example sentence for an word count exercise'
fdist = FreqDist()
for word in word_tokenize(sent):
    fdist[word.lower()] += 1
    
print(fdist.most_common(10))



import pandas as pd
from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
# print(positive_tweets)
print(type(positive_tweets))
print(positive_tweets[0]) 
# #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)

import os
import pandas as pd

path = r'Desktop'

# df = pd.DataFrame(positive_tweets)
# df.to_csv('filename.csv', index=False)

# df.to_csv(os.path.join(path,r'tweet_token_test1.csv'))



df = pd.read_excel(r'Desktop\1_Teacher_reviews_Q51_test.xlsx')
# print (df)

# print(df['Essay Text'])
teacher_reviews = df['Essay Text'].values.tolist()
# print(teacher_reviews)


print(df.head(5))

row_n = []

for i in range(len(df)):
    row_n.append(i)
    
df["row_num"] = row_n
print(df.head(5))
