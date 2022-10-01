 

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import sys
import argparse
import re
from unicodedata import normalize
import nltk
from nltk.stem import PorterStemmer
import datetime
import time 
import argparse 


# Building our classifier class. 
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
 
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes) 

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
 
df = pd.read_excel('polaridad.xlsx',
                skiprows = 1,
                names=['UID', 'texto', 'Sentiment'])
 
all_words = []
documents = []

stopwords_set = set(stopwords.words('spanish'))

def clean_stopwords(inp):
    stop_words = set(w.lower() for w in stopwords.words())
    out = ' '.join(filter(lambda x: x.lower() not in stopwords_set, inp.split()))
    return remove_special_chars(out)
 
def remove_special_chars(inp):
    out = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
        normalize( "NFD", inp), 0, re.I
    )
    return (out) 
 
for idx in df.index:
    texto= clean_stopwords(df.loc[idx, "texto"]) 
    print (texto)
    documents.append( (texto,df.loc[idx, "Sentiment"]) )
    words = word_tokenize(df.loc[idx, "texto"])
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] not in stopwords_set:
            all_words.append(w[0].lower())

 

 
save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
 
all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
 
random.shuffle(featuresets)
print(len(featuresets))
 

training_set, testing_set = train_test_split(featuresets,test_size = 0.1)
 
save_features = open("featuresets.pickle","wb")
joblib.dump(featuresets, save_features)
save_features.close()
 
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


save_classifier = open("originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

 
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

 
voted_classifier = VoteClassifier(
                                  classifier,
                                  LogisticRegression_classifier,
                                  BernoulliNB_classifier 
                                   )


