# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:12:38 2019

@author: Chirag
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re

dataset = pd.read_csv('AT&T_Data.csv')

dataset['Reviews'][0]

processed_review = []

review = re.sub('\'','',dataset['Reviews'][0])

for i in range(113):
    review = re.sub('\'','',dataset['Reviews'][i])
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(token) for token in review if not token in stopwords.words('english')]
    review = ' '.join(review)
    processed_review.append(review)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 7000)
X = cv.fit_transform(processed_review)
X = X.toarray()

temp = dataset['Label'].values
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
temp = lab.fit_transform(temp)
y = temp

print(cv.get_feature_names())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train, y_train)
print(n_b.score(X_train, y_train))
print(n_b.score(X_test, y_test))
print(n_b.score(X, y))
#As many times we run this code the score will change


y_pred= n_b.predict(X_test)

from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(precision_score(y_test,y_pred,average = 'micro'))
print(recall_score(y_test,y_pred,average = 'micro'))
print(f1_score(y_test,y_pred,average = 'micro'))

print(classification_report(y_test, y_pred))

