import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine_data.csv',
                      names = [
                               'Index',
                               'alcohol',
                               'malic acid',
                               'ash',
                               'alkalinity_of_ash',
                               'magnesium',
                               'total_phenols',
                               'flavanoids',
                               'nonflavanoid_phenols',
                               'proanthocyanins',
                               'colour_intensity',
                               'hue',
                               'diluted_wines',
                               'proline'                               
                              ]                     
                      )

X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y = dataset.iloc[: , 0].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.score(X, y)
y_predict = lin_reg.predict(X)
lin_reg.score(X, y)


x = np.array(range(177))
plt.scatter(x,y, label = 'original values')
plt.scatter(x,y_predict,c = 'r', label = 'prediction')
plt.legend()
plt.show()