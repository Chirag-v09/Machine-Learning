import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#-------naive based theorem----------(WORST)
from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()
n_b.fit(X_train,y_train)

# 90 %
n_b.score(X_train,y_train)
n_b.score(X_test,y_test)
n_b.score(X,y)

y_pred_nb = n_b.predict(X_test)

from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,confusion_matrix
cm = confusion_matrix(y_test,y_pred_nb)
print(precision_score(y_test,y_pred_nb,average = 'micro'))
print(recall_score(y_test,y_pred_nb,average = 'micro'))
print(f1_score(y_test,y_pred_nb,average = 'micro'))

print(classification_report(y_test, y_pred_nb))

#----------K Nearest Neighbors---------------(AVERAGE)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

# 91 %
knn.score(X_train,y_train)
knn.score(X_test,y_test)
knn.score(X,y)

y_pred_knn = knn.predict(X_test)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
cm_knn = confusion_matrix(y_test,y_pred_knn,)
print(precision_score(y_test, y_pred_knn,average = 'micro'))
print(recall_score(y_test, y_pred_knn, average = 'micro'))
print(f1_score(y_test, y_pred_knn, average = 'micro'))

print(classification_report(y_test, y_pred_knn))

#------------Logistic Regression------------(BEST)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

# 92 %
log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)
log_reg.score(X,y)

y_pred_log = log_reg.predict(X_test)

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
cm_log = confusion_matrix(y_test, y_pred_log)
print(precision_score(y_test, y_pred_log, average = 'micro'))
print(recall_score(y_test, y_pred_log , average = 'micro'))
print(f1_score(y_test, y_pred_log, average = 'micro'))

print(classification_report(y_test, y_pred_log))

#--------decison tree classifier-------
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 5)
dtf.fit(X_train,y_train)

dtf.score(X_train,y_train)
dtf.score(X_test,y_test)
dtf.score(X,y)

y_pred_dtf = dtf.predict(X_test)

#-----------------------------------------------
import graphviz
from sklearn import tree
dot = tree.export_graphviz(dtf, out_file="tree.dot", class_names = ["maligent", "benign"],
feature_names = dataset.feature_names, impurity = False, filled = True)

with open("tree.dot") as f:
    dot_graph = f.read()
    graphviz.Source(dot_graph)


graph = graphviz.Source(dot)
graph.render("Breast_cancer")