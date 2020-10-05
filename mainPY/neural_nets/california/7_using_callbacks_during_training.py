
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Let's load, split and scale the California housing dataset 
# (the original one, not the modified one as in chapter 2):
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# # Using Callbacks during Training

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])


def checkpointCallback(model):
    
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/neural_nets/california/7_checkpoint.h5", save_best_only=True)
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])
    model = keras.models.load_model("mainPY/neural_nets/california/7_checkpoint.h5") # rollback to best model
    mse_test = model.evaluate(X_test, y_test)


def earlyStoppingCallback(model):

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/neural_nets/california/7_earlyStopping.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                        restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb])
    mse_test = model.evaluate(X_test, y_test)


def validationTrainingRatioCallback(model):

    class PrintValTrainRatioCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    val_train_ratio_cb = PrintValTrainRatioCallback()
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[val_train_ratio_cb])


def tensorboardCallbackWithLogDir1(model):

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/neural_nets/california/7_tensorboardLogDir1.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb])


def tensorboardCallbackWithLogDir2(model):

    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.05))
    checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/neural_nets/california/7_tensorboardLogDir2.h5", save_best_only=True)
    tensorboard_cb = keras.callbacks.TensorBoard(get_run_logdir())
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb, tensorboard_cb])


checkpointCallback(model)
earlyStoppingCallback(model)
validationTrainingRatioCallback(model)
tensorboardCallbackWithLogDir1(model)
tensorboardCallbackWithLogDir2(model)


