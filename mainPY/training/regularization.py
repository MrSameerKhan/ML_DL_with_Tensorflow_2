import tensorflow as tf
from tensorflow import keras
import numpy as np
from functools import partial

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

tf.random.set_seed(42)
np.random.seed(42)

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

tf.random.set_seed(42)
np.random.seed(42)

# # Avoiding Overfitting Through Regularization
# ## l1 and l2 regularization

def layerSingle():

    layer = keras.layers.Dense(100, activation="elu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.01))
    # or l1(0.1) for ℓ1 regularization with a factor or 0.1
    # or l1_l2(0.1, 0.01) for both ℓ1 and ℓ2 regularization, with factors 0.1 and 0.01 respectively

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(300, activation="elu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dense(100, activation="elu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dense(10, activation="softmax",
                            kernel_regularizer=keras.regularizers.l2(0.01))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))


def layerRegular():

    RegularizedDense = partial(keras.layers.Dense,
                            activation="elu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.01))

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        RegularizedDense(300),
        RegularizedDense(100),
        RegularizedDense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))





def dropOut():

    # ## Dropout
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))



def alphaDropOut():

    # ## Alpha Dropout
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 20
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))

    model.evaluate(X_test_scaled, y_test)

    model.evaluate(X_train_scaled, y_train)

    history = model.fit(X_train_scaled, y_train)


    y_probas = np.stack([model(X_test_scaled, training=True)
                        for sample in range(100)])
    y_proba = y_probas.mean(axis=0)
    y_std = y_probas.std(axis=0)

    np.round(model.predict(X_test_scaled[:1]), 2)

    np.round(y_probas[:, :1], 2)

    np.round(y_proba[:1], 2)

    y_std = y_probas.std(axis=0)
    np.round(y_std[:1], 2)

    y_pred = np.argmax(y_proba, axis=1)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(accuracy)



def mcDropOut():

    # ## MC Dropout

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
    ])
    
    class MCDropout(keras.layers.Dropout):
        def call(self, inputs):
            return super().call(inputs, training=True)

    class MCAlphaDropout(keras.layers.AlphaDropout):
        def call(self, inputs):
            return super().call(inputs, training=True)

    mc_model = keras.models.Sequential([
        MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
        for layer in model.layers
    ])

    mc_model.summary()

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    mc_model.set_weights(model.get_weights())

    # Now we can use the model with MC Dropout:

    np.round(np.mean([mc_model.predict(X_test_scaled[:1]) for sample in range(100)], axis=0), 2)

    # ## Max norm
    layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                                kernel_constraint=keras.constraints.max_norm(1.))

    MaxNormDense = partial(keras.layers.Dense,
                            activation="selu", kernel_initializer="lecun_normal",
                            kernel_constraint=keras.constraints.max_norm(1.))

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        MaxNormDense(300),
        MaxNormDense(100),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=n_epochs,
                        validation_data=(X_valid_scaled, y_valid))


layerSingle()
layerRegular()
dropOut()
alphaDropOut()
mcDropOut()