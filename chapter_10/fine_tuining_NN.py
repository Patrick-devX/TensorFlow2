####################### Fine-Tuning neural Network Hyperparametres ###################

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import numpy as np
import os

housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)



def build_model(n_hidden=1, n_neurons=30, learning_rate=1e-3, input_shape=[8]):

    """
    create a simple Sequential model for univariate regression (only one output neueron),
    with the given input shape and the given number of hidden layers and neurons
    :param n_hidden:
    :param n_neurons:
    :param learning_rate:
    :param input_shape:
    :return:
    """
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=learning_rate))
    model.summary()

    return model

# Create a KerasRegressor based on this build_model() function
keras_regressor = keras.wrappers.scikit_learn.KerasRegressor(build_model)

#Fit
earlyStopping_cb= keras.callbacks.EarlyStopping(patience=10)
#keras_regressor.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                    #callbacks=[earlyStopping_cb])

#mse_test = keras_regressor.score(x_test, y_test)

#x_new = x_test[0:3]
#y_pred = keras_regressor.predict(x_new)

######### Many models variants #########
from  scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribution = { 'n_hidden': [0, 1, 2, 3],
                       'n_neurons': np.arange(1, 100),
                       'learning_rate': reciprocal(3e-4, 3e-2)
                       }

random_searchCV = RandomizedSearchCV(keras_regressor, param_distribution, n_iter=10, cv=3)
random_searchCV.fit(x_train, y_train, epochs=12, validation_data=(x_valid, y_valid),
                    callbacks=earlyStopping_cb)

# Get the best parameters
print(random_searchCV.best_params_)
print(random_searchCV.best_score_)


