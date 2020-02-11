import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


m=100
X = 6*np.random.randn(m,1)
y = 2*X**2 + X + 2 + 2*np.random.randn(m,1) 


plt.scatter(X,y)
plt.axis([-10,10, -10,100])
#        x-axis, y-axis  range
#that graph will show from which part to which part like we magnify that area
plt.show()

from sklearn.preprocessing import PolynomialFeatures
#PolynomialFeatures class is used to find the matrix which contains its squre value
#of the x column (i.e 1 column matrix)
poly = PolynomialFeatures(degree = 2, include_bias = False)
# include_bais will add a new column in the X_poly matrix in begning having value 1
X_poly = poly.fit_transform(X)


#Through this we train the model i.e now lin_reg object contains the equation(or formula) 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)

lin_reg.score(X_poly,y)


#Testing on new data on that same formula
X_new = np.linspace(-5,5,50).reshape(-1,1)
X_new_poly = poly.fit_transform(X_new)

y_new = lin_reg.predict(X_new_poly)


#Now we plot them together to see how or model predict
plt.scatter(X,y)
plt.plot(X_new,y_new,c = 'r')
plt.axis([-5,5, -5,50])
plt.show()

lin_reg.predict([[-5,25]])

lin_reg.coef_
lin_reg.intercept_

a = 2*25 -5 +2  #for x =-5 prediction by manually
