
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

lin_reg.score(X,y)
lin_reg.predict([[0.00632, 0, 2.21, 0, 0.538, 6.675, 75.6, 5.05, 2, 311, 15.5, 390, 10]])

y_pred = lin_reg.predict(X)

#------------------------

x=np.array(range(506))
plt.scatter(x,y,c='r', label = 'original data')
plt.scatter(x,y_pred,c='g', label = 'line of prediction')
plt.legend()
plt.show()
