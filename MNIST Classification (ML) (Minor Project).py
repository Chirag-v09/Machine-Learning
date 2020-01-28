# Importing the required Libraries for the code:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
from sklearn.datasets import load_digits
dataset = load_digits()

# Feature matrix and vector of prediction
X = dataset.data
y = dataset.target

# Dividing the dataset in train & test dataset
from  sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2)

# Visualizing the dataset
some_digit = X[501]
some_digit_image = some_digit.reshape(8,8)
plt.imshow(some_digit_image)
plt.show()

# Importing the Classifier Algorithm DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 9)
dtf.fit(X_train,y_train)

# Checking the scores
print(dtf.score(X_train,y_train))
print(dtf.score(X_test,y_test))
print(dtf.score(X,y))

# Random prediction
print(dtf.predict(X[[112,456,1256,500,1000],0:64]))

# prediction of the test dataset
y_pred = dtf.predict(X_test)

# Seeing the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# PERFOEMANCE MATRIX:-
# Seeing the Precision Score, Recall Score, F1 Score
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(y_test, y_pred,average = 'micro'))
print(recall_score(y_test, y_pred,average = 'micro'))
print(f1_score(y_test, y_pred, average = 'micro'))

# Confusion Matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
