import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.6)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans

wcv = []

for i in range(1, 16):
    km = KMeans(n_clusters = i)
    km.fit(x)
    wcv.append(km.inertia_)
    # In inertia it's a tuple where the wsv formula is stored and find out it's value through the formula


plt.plot(range(1, 16), wcv)
plt.show()

km = KMeans(n_clusters = 5)
y_pred = km.fit_predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1])
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1])
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1])
plt.show()