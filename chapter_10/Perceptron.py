import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]  # Petal Length, Petal Width
Y = (iris.target == 0).astype('int32')  # Iris setosa

per_clf = Perceptron()
per_clf.fit(X, Y)

y_pred = per_clf.predict([[2, 0.5]])

print(y_pred)



