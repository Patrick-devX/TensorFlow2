
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Using Kera to load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
print(x_train_full.shape)
print(x_train_full.dtype)

x_valid, x_train = x_train_full[:5000]/255.0, x_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#Class Names
class_names = ["T-shirttop", "Trouser", "Pullover", "Dress", "Coat", "Scandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(y_train[0])

#Creating the model using the sequential API

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  #Convert each imput Image into 1D array
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

#Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

# Get All parameters of layer
model.layers
hidden_layer1 = model.layers[1]
hidden_layer1.name
model.get_layer('dense') is hidden_layer1
weights, biases = hidden_layer1.get_weights()
weights.shape
biases.shape

#Evaluate the Model
model.evaluate(x_test, y_test)

#Using the model to make predictions
x_new = x_test[:3]
y_proba = model.predict(x_new)
y_proba = y_proba.round(2)
print(y_proba)


y_pred = model.predict(x_new)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)




