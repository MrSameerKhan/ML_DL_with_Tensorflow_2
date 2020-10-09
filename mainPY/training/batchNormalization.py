import tensorflow as tf
from tensorflow import keras
import numpy as np

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

tf.random.set_seed(42)
np.random.seed(42)


def batchNormalization():

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
    ])

    model.summary()

    bn1 = model.layers[1]
    [(var.name, var.trainable) for var in bn1.variables]

    bn1.updates

    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid))


def BNbeforeActivationFunction():
    # Sometimes applying BN before the activation function works 
    # better (there's a debate on this topic). Moreover, 
    # the layer before a `BatchNormalization` layer 
    # does not need to have bias terms, since the 
    # `BatchNormalization` layer some as well, 
    # it would be a waste of parameters, so you can 
    # set `use_bias=False` when creating those layers:

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid))


batchNormalization()
BNbeforeActivationFunction()