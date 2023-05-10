

#There are many use cases in which you may want to have multiple outputs: See Page 406

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


input_layer = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_layer)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

concat = keras.layers.Concatenate()[input_layer, hidden2]

output = keras.layers.Dense(1, name='main_output')(concat)
aux_output = keras.layers.Dense(1, name='aux_output')(hidden2)

model = keras.Model(inputs=[input_layer], output=[output])

#if you want to send a subset of the features through the wide path
#and a different subset (possibly overlapping) through the deep path
# In this case, one solution is to use multiple inputs.

#For example, suppose we want to send five features through the wide path
#(features 0 to 4), and six features through the deep path (features 2 to 7):

input_layer_A = keras.layers.Input(shape=x_train[5])
input_layer_B = keras.layers.Input(shape=x_train[6])

hidden1 = keras.layers.Dense(30, activation='relu')(input_layer_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

concat = keras.layers.Concatenate()[input_layer_A, hidden2]
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_layer_A, input_layer_B], output=[output, aux_output])
#Compile
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd')

x_train_A, x_train_B = x_train[:, :5], x_train[:, 2:]
x_valid_A, x_valid_B = x_valid[:, :5], x_valid[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]

x_new_A = x_test_A[0:3]
x_new_B = x_test_B[0:3]

history = model.fit([x_train_A, x_train_B], [y_train, y_train], epochs=20,
                    validation_data=([x_valid_A, x_valid_B], [y_valid, y_valid]))

total_loss, main_loss, aux_loss = model.evaluate([x_test_A, x_test_B], [y_test, y_test])

y_pred_main, y_pred_aux = model.predict([x_new_A, x_new_B])


############### Saving and Restoring a Model ################

model.save('my_keras_model.h5')
#Load model
model = keras.models.load_model('my_kears_model.h5')