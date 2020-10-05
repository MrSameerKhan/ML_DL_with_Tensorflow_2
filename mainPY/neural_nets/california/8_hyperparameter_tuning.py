
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Let's load, split and scale the California housing dataset 
# (the original one, not the modified one as in chapter 2):
housing = fetch_california_housing()

def hyper_tuning(housing):

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    keras_reg.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    mse_test = keras_reg.score(X_test, y_test)
    X_new = X_test[:3]
    y_pred = keras_reg.predict(X_new)


# **Warning**: the following cell crashes at the end of training. This seems to be caused by [Keras issue #13586](https://github.com/keras-team/keras/issues/13586), which was triggered by a recent change in Scikit-Learn. [Pull Request #13598](https://github.com/keras-team/keras/pull/13598) seems to fix the issue, so this problem should be resolved soon.

def randomized(housing):


    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation="relu"))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

    np.random.seed(42)
    tf.random.set_seed(42)

    from scipy.stats import reciprocal
    from sklearn.model_selection import RandomizedSearchCV

    param_distribs = {
        "n_hidden": [0, 1, 2, 3],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
    rnd_search_cv.fit(X_train, y_train, epochs=100,
                        validation_data=(X_valid, y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    print(rnd_search_cv.best_params_)
    print(rnd_search_cv.best_score_)
    print(rnd_search_cv.best_estimator_)
    print(rnd_search_cv.score(X_test, y_test))

    model = rnd_search_cv.best_estimator_.model
    print(model)
    print(model.evaluate(X_test, y_test))

# hyper_tuning(housing)
randomized(housing)