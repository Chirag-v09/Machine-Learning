# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:28:45 2019

@author: Chirag
"""
# Machine learning


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.externals import joblib

dataset = pd.read_csv('adult sal.csv', names = ['age',
                                          'workclass',
                                          'fnlwgt',
                                          'education',
                                          'education-num',
                                          'marital-status',
                                          'occupation',
                                          'relationship',
                                          'race',
                                          'gender',
                                          'capital-gain',
                                          'capital-loss',
                                          'hours-per-week',
                                          'native-country',
                                          'salary'],
                        na_values = ' ?')

X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, -1].values
test = pd.DataFrame(X[:, [1, 3, 5, 6, 7, 8, 9, 13]])

test[0].value_counts()
test[1].value_counts()
test[2].value_counts()
test[3].value_counts()
test[4].value_counts()
test[5].value_counts()
test[6].value_counts()
test[7].value_counts()


test[0] = test[0].fillna(' Private')
test[1] = test[1].fillna(' HS-grad')
test[2] = test[2].fillna(' Married-civ-spouse')
test[3] = test[3].fillna(' Prof-specialty')
test[4] = test[4].fillna(' Husband')
test[5] = test[5].fillna(' White')
test[6] = test[6].fillna(' Male')
test[7] = test[7].fillna(' United-States')


X[:, [1, 3, 5, 6, 7, 8, 9, 13]] = test

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:, 1] = lab.fit_transform(X[:, 1].astype(str))
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 5] = lab.fit_transform(X[:, 5])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 9] = lab.fit_transform(X[:, 9])
X[:, 13] = lab.fit_transform(X[:, 13])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1, 3, 5, 6, 7, 8, 9, 13])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y = lab.fit_transform(y)
lab.classes_

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# =============================================================================
# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(X_train, y_train)
# joblib.dump(svm, "svm.pkl")
# =============================================================================
svm = joblib.load("svm.pkl")

svm.score(X_train, y_train)#86.3 --- 85.8 --- 86.0
svm.score(X_test, y_test)#84. --- 86.2
svm.score(X, y)#85.9 --- 85.9  No Change


# ============================================================================= 
# from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)
# joblib.dump(log_reg, "log_reg.pkl")
# =============================================================================
log_reg = joblib.load("log_reg.pkl")

log_reg.score(X_train, y_train)#85.4 --- 85.0 --- 85.2
log_reg.score(X_test, y_test)#84.4 --- 85.6
log_reg.score(X, y)#85.1 --- 85.6


# =============================================================================
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# joblib.dump(knn, "knn.pkl")
# =============================================================================
knn = joblib.load("knn.pkl")

print(knn.score(X_train, y_train))#87.5 --- 86.0 --- 86.2
print(knn.score(X_test, y_test))#82.3 --- 86.8
print(knn.score(X, y))#86.2 --- 86.2  No Change


# =============================================================================
# from sklearn.tree import DecisionTreeClassifier
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# joblib.dump(dtc, "dtc.pkl")
# =============================================================================
dtc = joblib.load("dtc.pkl")

print(dtc.score(X_train, y_train))#99.9 --- 95.2 --- 95.2
print(dtc.score(X_test, y_test))#80.9 --- 95.2
print(dtc.score(X, y))#95.2 --- 95.2  No Change


# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# n_b = GaussianNB()
# n_b.fit(X_train, y_train)
# joblib.dump(n_b, "naive.pkl")
# =============================================================================
n_b = joblib.load("naive.pkl")

print(n_b.score(X_train, y_train))#51.0 --- 50.7 --- 51.0
print(n_b.score(X_test, y_test))#50.1 --- 51.0
print(n_b.score(X, y))#50.8 --- 50.8  No Change


#----------Ensenbling----------

# #---------Voting---------------
# =============================================================================
# from sklearn.ensemble import VotingClassifier
# vot1 = VotingClassifier([('LR', log_reg),
#                        ('KNN', knn),
#                        ('SVM', svm)], voting ='hard')
# vot1.fit(X_train, y_train)
# joblib.dump(vot1, "Votting_LR_KNN_SVM.pkl")
# =============================================================================
vot1 = joblib.load("Votting_LR_KNN_SVM.pkl")

print(vot1.score(X_train, y_train))#86.7 --- 86.3 --- 86.3
print(vot1.score(X_test, y_test))#84.6
print(vot1.score(X, y))#86.2

#---------------------
# =============================================================================
# from sklearn.ensemble import VotingClassifier
# vot = VotingClassifier([('LR', log_reg),
#                        ('KNN', knn),
#                        ('NB', n_b)], voting ='soft')
# vot.fit(X_train, y_train)
# joblib.dump(vot, "Votting_LR_KNN_NB.pkl")
# =============================================================================
vot = joblib.load("Votting_LR_KNN_NB.pkl")

print(vot.score(X_train, y_train))#87.0 --- 86.2
print(vot.score(X_test, y_test))#83.5
print(vot.score(X, y))#86.1

#----------Bagging----------
# =============================================================================
# from sklearn.ensemble import BaggingClassifier
# bag = BaggingClassifier(knn, n_estimators = 5)
# bag.fit(X_train, y_train)
# joblib.dump(bag, "bag_knn.pkl")
# =============================================================================
bag = joblib.load("bag_knn.pkl")

print(bag.score(X_train, y_train))#87.4 --- 85.9
print(bag.score(X_test, y_test))#81.6
print(bag.score(X, y))#86

#---------Bagging of Decision Tree--------------Random Forest------------------
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier
# rf =  RandomForestClassifier()
# rf.fit(X_train, y_train)
# joblib.dump(rf, "RandomForest.pkl")
# =============================================================================
rf = joblib.load("RandomForest.pkl")

print(rf.score(X_train, y_train))#98.7 --- 95.0
print(rf.score(X_test, y_test))#84.2
print(rf.score(X, y))#95.1


#-------------Grid Search----------------

param_grid = {'n_neighbors' : [1,2,3,4,5,6,7,8,9] } #** [{'n_neighbors' : [1,2,3,4,5,6,7,8,9]}]

param_grid1 = [{'criterion' :['gini','entropy'] },
               {'max_depth' :[3,4,5,6,7,8,9] }
              ]

# =============================================================================
# from sklearn.model_selection import GridSearchCV # cv - count vector (no. of instances)
# grid = GridSearchCV(knn, param_grid)
# %time grid.fit(X, y)#53m 20s --- 3h 54m 7s
# joblib.dump(grid, "GridSearch knn.pkl")
# =============================================================================
grid = joblib.load("GridSearch knn.pkl")

print("Best Estimator:- ",grid.best_estimator_)
print("Best Index:- ",grid.best_index_)
print("Best Parameter:- ",grid.best_params_)
print("Best Score:- ",grid.best_score_)


# =============================================================================
# grid1 = GridSearchCV(dtc, param_grid1)
# %time grid1.fit(X, y)# 7.5s
# joblib.dump(grid1, "GridSearch dtc.pkl")
# =============================================================================
grid1 = joblib.load("GridSearch dtc.pkl")

print("Best Estimator:- ",grid1.best_estimator_)
print("Best Index:- ",grid1.best_index_)
print("Best Parameter:- ",grid1.best_params_)
print("Best Score:- ",grid1.best_score_)
# max-depth = 8


#-----------------Randomized Search-----------------------

# =============================================================================
# from sklearn.model_selection import RandomizedSearchCV
# rand = RandomizedSearchCV(knn, param_distributions = param_grid)
# %time rand.fit(X_train, y_train)#30m 29s
# joblib.dump(rand, "RandomizedSearch knn.pkl")
# =============================================================================
rand = joblib.load("RandomizedSearch knn.pkl")

print(rand.best_estimator_)
print(rand.best_index_)
print(rand.best_params_)
print(rand.best_score_)


param_grid2 = {'criterion' :['gini','entropy'],
               'max_depth' :[3,4,5,6,7,8,9] 
              }

# =============================================================================
# from sklearn.model_selection import RandomizedSearchCV
# rand1 = RandomizedSearchCV(dtc, param_distributions = param_grid2)
# %time rand1.fit(X_train, y_train)#5.2s
# joblib.dump(rand1, "RandomizedSearch dtc.pkl")
# =============================================================================
rand1 = joblib.load("RandomizedSearch dtc.pkl")

print(rand1.best_estimator_)
print(rand1.best_index_)
print(rand1.best_params_)
print(rand1.best_score_)




