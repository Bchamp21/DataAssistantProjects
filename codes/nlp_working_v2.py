# Ref: https://shanikaperera11.medium.com/positive-or-negative-spam-or-not-spam-a-simple-text-classification-problem-using-python-727efd64c238

import pandas as pd
from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
print("type of positive tweets:",type(positive_tweets))
print("******************************************************")
print(type(twitter_samples))
print("twitter_samples",twitter_samples)
print("******************************************************")
print("tweet_tokens:",tweet_tokens)
print("******************************************************")
print("type of tweet_tokens:",type(tweet_tokens))
##Output: ['#FollowFriday', '@France_Inte', '@PKuchly57', '@Milipol_Paris', 'for', 'being', 'top', 'engaged', 'members', 'in', 'my', 'community', 'this', 'week', ':)']
# print(twitter_samples.tokenized('positive_tweets.json')[1])
# print(twitter_samples.tokenized('positive_tweets.json')[2])




from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence
print("******************************************************")
print("Lemmatized sentance tweet_tokens:",lemmatize_sentence(tweet_tokens))
##Output: ['#FollowFriday', '@France_Inte', '@PKuchly57', '@Milipol_Paris', 'for', 'be', 'top', 'engage', 'member', 'in', 'my', 'community', 'this', 'week', ':)']



import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens



from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print("******************************************************")
print("Remove noise in tweet_tokens:",remove_noise(tweet_tokens, stop_words))

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
# print(positive_tweet_tokens[500])
print("******************************************************")
print("Positive tweet_tokens at index 5:",positive_tweet_tokens[5])
print("Postive cleaned tokens list",positive_cleaned_tokens_list[2])



def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print("Type of freq dist", type(freq_dist_pos))
print("freq dist", freq_dist_pos)
print("******************************************************")
print("Type of freq dist", type(freq_dist_pos.most_common(20)))
print("Freq dist:",freq_dist_pos.most_common(20))


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


import random

positive_dataset = [(tweet_dict, "Positive")
                      for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                      for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]



from nltk import classify
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

classifier = NaiveBayesClassifier.train(train_data)

print("******************************************************")
print("Accuracy is:", classify.accuracy(classifier, test_data))

print("******************************************************")
print("Classifier show most informative features:",classifier.show_most_informative_features(10))

df_c = type(classifier.show_most_informative_features(10))

print("******************************************************")
print(df_c)


def show_most_informative_features_in_list(classifier, n=10):
    """
    Return a nested list of the "most informative" features 
    used by the classifier along with it's predominant labels
    """
    cpdist = classifier._feature_probdist       # probability distribution for feature values given labels
    feature_list = []
    for (fname, fval) in classifier.most_informative_features(n):
        def labelprob(l):
            return cpdist[l, fname].prob(fval)
        labels = sorted([l for l in classifier._labels if fval in cpdist[l, fname].samples()], 
                        key=labelprob)
        feature_list.append([fname, labels[-1],fval])
    return feature_list


print("******************************************************")
print("show most informative features function in list:",show_most_informative_features_in_list(classifier, 10))

df_s = []
print("******************************************************")
print("Type of informative features",type(show_most_informative_features_in_list(classifier)))
# df_s["show_most_informative_feature"] = type(show_most_informative_features_in_list(classifier))
print("******************************************************")
print(df_s)

print("******************************************************")
print(classifier.show_most_informative_features(10),show_most_informative_features_in_list(classifier))
print("******************************************************")
print("Just 1 element",show_most_informative_features_in_list(classifier)[0])

# output_path = r'C:\Users\TK20\Desktop\Solution'
# output_filename = r'1_teacher_review_sentiment_Q51.csv'
# review_col_name = 'Essay Text'

# df.to_csv(os.path.join(output_path,output_filename))








from nltk.tokenize import word_tokenize

custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print("******************************************************")
print("Classifier classify:",classifier.classify(dict([token, True] for token in custom_tokens)))


custom_tweet = "Meeting the needs of students who do not yet qualify for an IEP, but who are unable to complete work on their own in my room. I implement interventions and strategies discussed with the problem solving team."

custom_tokens = remove_noise(word_tokenize(custom_tweet))
print("******************************************************")
print("Classifier classify:",classifier.classify(dict([token, True] for token in custom_tokens)))

print("******************************************************")
print("Top ten most informative words: ")

for item in classifier.most_informative_features()[:10]:
    print("item ",item[0])
    
#Sample input reviews
input_reviews = [
    "Started off as the greatest series of all time, but had the worst ending of all time.",
    "Exquisite. 'Big Little Lies' takes us to an incredible journey with its emotional and intriguing storyline.",
    "I love Brooklyn 99 so much. It has the best crew ever!!",
    "The Big Bang Theory and to me it's one of the best written sitcoms currently on network TV.",
    "'Friends' is simply the best series ever aired. The acting is amazing.",
    "SUITS is smart, sassy, clever, sophisticated, timely and immensely entertaining!",
    "Cumberbatch is a fantastic choice for Sherlock Holmes-he is physically right (he fits the traditional reading of the character) and he is a damn good actor",
    "What sounds like a typical agent hunting serial killer, surprises with great characters, surprising turning points and amazing cast."
    "This is one of the most magical things I have ever had the fortune of viewing.",
    "I don't recommend watching this at all!"
]
print("******************************************************")
print("Input reviews:",input_reviews)
print("******************************************************")
print("Type of input reviews:",type(input_reviews))
print("******************************************************")
print("Predictions: ")

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print("Predicted sentiment: ", pred_sentiment)
    print("Probability: ", round(probdist.prob(pred_sentiment), 2))






"**********************************************************************"
"***********************************"
"Function for Sentiment Analysis"
"***********************************"


def sentiment_reviews(review_list):
    Predicted_sentiment = []
    Probability = []
    def extract_features(word_list):
        return dict([(word, True) for word in word_list])

    
    for review in review_list:
        print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment: ", pred_sentiment)
        print("Probability: ", round(probdist.prob(pred_sentiment), 2))
        Predicted_sentiment.append(pred_sentiment)
        Probability.append(round(probdist.prob(pred_sentiment), 2))
     
    return Predicted_sentiment, Probability






print("**************************************************")
print("1 Predictions for Teacher Reviews: Q51")
print("**************************************************")


import os

input_path_filename = r'C:\Users\TK20\Desktop\1_Teacher_reviews_Q51_test.xlsx'
output_path = r'C:\Users\TK20\Desktop\Solution'
output_filename = r'1_teacher_review_sentiment_Q51.csv'
review_col_name = 'Essay Text'

df = pd.read_excel(input_path_filename)

teacher_reviews = df[review_col_name].values.tolist()
print("Predictions for Teacher Reviews: ")
df["Predicted sentiment"],df["Probability"] = sentiment_reviews(teacher_reviews)
# df.to_csv(os.path.join(output_path,output_filename), index=False)

fdist = FreqDist()
for sent in teacher_reviews:    
    for word in word_tokenize(sent):
        fdist[word.lower()] += 1

print(fdist.most_common(50))
common = fdist.most_common()

output_filename = r'freq_dist_Q51.csv'

df1 = pd.DataFrame(common, columns=['word', 'count'])
print(list(df1['word'].head(10)))
df1["Predicted sentiment"],df1["Probability"] = sentiment_reviews(list(df1['word']))
# df1.to_csv(os.path.join(output_path,output_filename), index=False)

df_Q51 = df1


filt1 = df[review_col_name].str.contains('needs')
filt2 = df[review_col_name].str.contains('class')
print(df[filt1])
print(df[filt2])
output_filename = r'needs_Q51.csv'
df[filt1].to_csv(os.path.join(output_path,output_filename), index=False)
output_filename = r'class_Q51.csv'
df[filt2].to_csv(os.path.join(output_path,output_filename), index=False)


print("**************************************************")
print("2 Teachers principals reviews test: Q40")
print("**************************************************")


import os

input_path_filename = r'C:\Users\TK20\Desktop\2_Teachers_principals_reviews_Q40_test.xlsx'
output_path = r'C:\Users\TK20\Desktop\Solution'
output_filename = r'2_Teachers_sentiment_Q40.csv'
review_col_name = 'Essay Text'

df = pd.read_excel(input_path_filename)

teacher_reviews = df[review_col_name].values.tolist()

print("Predictions for Teacher Reviews: ")
df["Predicted sentiment"],df["Probability"] = sentiment_reviews(teacher_reviews)
print(df.head(5))


fdist = FreqDist()
for sent in teacher_reviews:    
    for word in word_tokenize(sent):
        fdist[word.lower()] += 1

print(fdist.most_common(50))
common = fdist.most_common()

output_filename = r'freq_dist_Q40.csv'

df1 = pd.DataFrame(common, columns=['word', 'count'])
print(list(df1['word'].head(10)))
df1["Predicted sentiment"],df1["Probability"] = sentiment_reviews(list(df1['word']))
# df1.to_csv(os.path.join(output_path,output_filename), index=False)

df_Q40 = df1

## Common words between both the dfs
list1_as_set = set(df_Q51['word'])
intersection = list1_as_set.intersection(df_Q40['word'])
# Find common elements of set and list

intersection_as_list = list(intersection)

print(intersection_as_list)


y1 = df_Q51[df_Q51['word'].isin(intersection_as_list)]

print("y1:",y1) ## 136 common

y2 = df_Q40[df_Q40['word'].isin(intersection_as_list)]

print("y2:",y2) ## 136 common

y_merged = pd.merge(y1,y2,how='inner',left_on=['word'],right_on=['word'], suffixes=("_Q51","_Q40"))

print(y_merged)
output_filename = r'Common_words_Q51_Q40.csv'
# y_merged.to_csv(os.path.join(output_path,output_filename), index=False)


filt1 = df[review_col_name].str.contains('needs')
filt2 = df[review_col_name].str.contains('class')
print(df[filt1])
print(df[filt2])
output_filename = r'needs_Q40.csv'
df[filt1].to_csv(os.path.join(output_path,output_filename), index=False)
output_filename = r'class_Q40.csv'
df[filt2].to_csv(os.path.join(output_path,output_filename), index=False)
