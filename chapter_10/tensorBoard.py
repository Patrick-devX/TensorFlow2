from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import os
import numpy as np

housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)


#For example, suppose we want to send five features through the wide path
#(features 0 to 4), and six features through the deep path (features 2 to 7):

# Model
input_layer_A = keras.layers.Input(shape=[5], name='wide_input')
input_layer_B = keras.layers.Input(shape=[6], name='deep_input')

hidden1 = keras.layers.Dense(30, activation='relu')(input_layer_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)

concat = keras.layers.concatenate([input_layer_A, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_layer_A, input_layer_B], outputs=[output])
model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=1e-3))



x_train_A, x_train_B = x_train[:, :5], x_train[:, 2:]
x_valid_A, x_valid_B = x_valid[:, :5], x_valid[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]
x_new_A = x_test_A[0:3]
x_new_B = x_test_B[0:3]

#defining a root log directory
root_logdir = os.path.join(os.curdir, '../my_logs')

def get_run_logdir():
    """
    generate sub directory path based on the current data and time so that it s different an every run
    :return: the generated sub directory path
    """
    import time
    #build log id
    run_id = time.strftime('run_%y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_sub_logdir = get_run_logdir()

#TensorBoard callback
tensorboard_cb = keras.callbacks.TensorBoard(run_sub_logdir)

history = model.fit((x_train_A, x_train_B), y_train, epochs=100,
                    validation_data=((x_valid_A, x_valid_B), y_valid), callbacks=[tensorboard_cb])

mse_test = model.evaluate((x_test_A, x_test_B), y_test)
y_pred = model.predict((x_new_A, x_new_B))

#TensorBoard set up
# 1 Create your directory where the logs have to be saved
# 2 Create the tensorBoard callback and specifie the lod directory in it
# 3 call the callback in the fit() method
# 4 Got to the termonal and activate the virtual environement variable: .\venv\Scripts\activate
# 5 Go back to the project directory myproject\mylog
# cal: tensorboard --logdir=./my_logs --port=6006