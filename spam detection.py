#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("spam.csv", encoding = "latin-1")

dataset_labels = dataset['v1']

dataset = dataset.drop(columns = ['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'])

dataset.head()

df_ham = dataset[dataset.v1 == "ham"]
df_spam = dataset[dataset.v1 == 'spam']

df_ham.head()

df_spam.head()

X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels, )

#extracting n-grams from the text data
countvect  = CountVectorizer(ngram_range=(2,2))
x_counts = countvect.fit(X_train.v2)
#preparing for training set
x_train_df = countvect.transform(X_train.v2)

#preparing for test set
x_test_df = countvect.transform(X_test.v2)

#now applying machine learning model
clf = MultinomialNB()

clf.fit(x_train_df, y_train)

predict = clf.predict(x_test_df)

acc = accuracy_score(y_test, predict)

print(acc)

