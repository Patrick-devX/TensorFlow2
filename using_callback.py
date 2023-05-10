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

############### Using Callbacks ################################
##### ModelCheckpoint #####
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_regression_model.h5")
#history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid,y_valid), callbacks=[checkpoint_cb])

# Moreover, if you use a validation set during training, you can set save_best_only = True when creating the ModelCheckpoint.
# In this Catse, it will only save your model when its performance on the validation set is best so far.
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_regression_model.h5", save_best_only=True)
#history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid,y_valid), callbacks=[checkpoint_cb])
#model = keras.models.load_model("my_keras_regression_model.h5")

##### EarlyStopping #####
# patience: number of epohcs oc stopping
early_stopping_cb = keras.callbacks.EarlyStopping (patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid,y_valid), callbacks=[checkpoint_cb, early_stopping_cb])

####### Own callback function for more control ########

class myCallback (keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.9):
            print("\nReached 90% accuracy so cancelling training ! ")
            self.model.stop_training = True

callbacks = myCallback() # Use it ater in fit function



mse_test = model.evaluate(x_test,y_test)
x_new = x_test[0:3]
y_pred = model.predict(x_new)

print(housing.feature_names)
print(x_new)
###############
print(y_pred)