# Import dataset
from sklearn.datasets import load_iris
dataset = load_iris()

# Feature matrix and vector of prediction
X = dataset.data
y = dataset.target

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_new = pca.fit_transform(X)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log_1 = LogisticRegression()

log.fit(X, y)
log_1.fit(X_new, y)

print(log.score(X, y))
print(log_1.score(X_new, y))


# Manual PCA done as we know that the petal length and petal width
# are the important features for getting the prediction.

X_1 = dataset.data[:, 2:4]

log_2 = LogisticRegression()
log_2.fit(X_1, y)
print(log_2.score(X_1, y))

