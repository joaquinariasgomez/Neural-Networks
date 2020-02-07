import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

#print(X)
print(y)

per_clt = Perceptron()
per_clt.fit(X, y)

y_pred = per_clt.predict([[2, 0.5]])
print(y_pred)
