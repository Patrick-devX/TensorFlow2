from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import numpy as np

housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]), # x_train.shape = (11610, 8), x_train.shape[1:] = (8,)
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='sgd')
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid,y_valid))
mse_test = model.evaluate(x_test,y_test)
x_new = x_test[0:3]
y_pred = model.predict(x_new)

print(housing.feature_names)
print(x_new)
###############
print(y_pred)