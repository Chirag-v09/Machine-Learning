import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset=load_iris()

X=dataset.data
y=dataset.target

# Much better to classify between there flowers
plt.scatter(X[y==0,2] ,X[y==0,3],c='r',label="sedosa")
plt.scatter(X[y==1,2] ,X[y==1,3],c='b',label="vericolor")
plt.scatter(X[y==2,2] ,X[y==2,3],c='g',label="virginica")

plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("Analysis")
plt.legend()
plt.show()


# Not much better to classify between there flowers
plt.scatter(X[y==0,0] ,X[y==0,1], c='r', label="sedosa")
plt.scatter(X[y==1,0] ,X[y==1,1], c='b', label="vericolor")
plt.scatter(X[y==2,0] ,X[y==2,1], c='g', label="virginica")

plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("Analysis")
plt.legend()
plt.show()
