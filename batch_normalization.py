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

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()

# The Batch Normalization scales and shifts inputs: The Batch Normalization Layer include thereby one offset
# parameter per input. This parameter (the bias) from the previous layer can be removed (jus pass use_bias=False when creating it)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu')
    keras.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu')
    keras.layers.Dense(10, activation="softmax")
])
model.summary()