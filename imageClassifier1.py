
import tensorflow as tf
from tensorflow import keras

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

history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

# Get All parameters of layer

model.layers
hidden_layer1 = model.layers[1]
hidden_layer1.name
model.get_layer('dense') is hidden_layer1
weights, biases = hidden_layer1.get_weights()
weights.shape
biases.shape

#Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])



