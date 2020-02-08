import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

print(y)
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=y)

per_clt = Perceptron()
per_clt.fit(X, y)

prediction = [1, 0.25]

y_pred = per_clt.predict([prediction])

plt.scatter(prediction[0], prediction[1], c='b')
print(y_pred.tolist()[0])
plt.show()
