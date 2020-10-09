
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time


# ## Custom loss function
# Let's start by loading and preparing the 
# California housing dataset. We first load it,
#  then split it into a training set, a validation set 
# and a test set, and finally we scale it:

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

input_shape = X_train.shape[1:]

def customLossSaveWeights():
    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", 
    kernel_initializer="lecun_normal",input_shape=input_shape),
    keras.layers.Dense(1),
    ])

    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=8,
                validation_data=(X_valid_scaled, y_valid))

    # ## Saving/Loading Models with Custom Objects
    model.save("mainPY/custom/my_model_with_a_custom_loss.h5")


def customLossLoadWeights():

    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", 
    kernel_initializer="lecun_normal",input_shape=input_shape),
    keras.layers.Dense(1),
    ])

    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
    model.save("mainPY/custom/my_model_with_a_custom_loss.h5")
    model = keras.models.load_model(
        "mainPY/custom/my_model_with_a_custom_loss.h5",
        custom_objects={"huber_fn": huber_fn})

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))


def customLossSaveThreshold_2():

    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", 
    kernel_initializer="lecun_normal",input_shape=input_shape),
    keras.layers.Dense(1),
    ])

    model.compile(loss=create_huber(2.0), 
    optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.save("mainPY/custom/my_model_with_a_custom_loss_threshold_2.h5")


def customLossLoadThreshold_2():

    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", 
    kernel_initializer="lecun_normal",input_shape=input_shape),
    keras.layers.Dense(1),
    ])

    model.compile(loss=create_huber(2.0), 
    optimizer="nadam", metrics=["mae"])

    model = keras.models.load_model("mainPY/custom/my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))


def customLossWithClassSave():
    class HuberLoss(keras.losses.Loss):
        def __init__(self, threshold=1.0, **kwargs):
            self.threshold = threshold
            super().__init__(**kwargs)
        def call(self, y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < self.threshold
            squared_loss = tf.square(error) / 2
            linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])
    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))

    model.save("mainPY/custom/my_model_with_a_custom_loss_class.h5")


def customLossWithClassLoad():

    class HuberLoss(keras.losses.Loss):
        def __init__(self, threshold=1.0, **kwargs):
            self.threshold = threshold
            super().__init__(**kwargs)
        def call(self, y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < self.threshold
            squared_loss = tf.square(error) / 2
            linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])
    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
    model = keras.models.load_model("mainPY/custom/my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))


def customFunctionsSave():

    def my_softplus(z): # return value is just tf.nn.softplus(z)
        return tf.math.log(tf.exp(z) + 1.0)

    def my_glorot_initializer(shape, dtype=tf.float32):
        stddev = tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)

    def my_l1_regularizer(weights):
        return tf.reduce_sum(tf.abs(0.01 * weights))

    def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
        return tf.where(weights < 0., tf.zeros_like(weights), weights)

    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1, activation=my_softplus,
                        kernel_regularizer=my_l1_regularizer,
                        kernel_constraint=my_positive_weights,
                        kernel_initializer=my_glorot_initializer),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.save("mainPY/custom/my_model_with_many_custom_parts.h5")

def customFunctionsClassL1Regularization():

    def my_softplus(z): # return value is just tf.nn.softplus(z)
        return tf.math.log(tf.exp(z) + 1.0)

    def my_glorot_initializer(shape, dtype=tf.float32):
        stddev = tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)

    def my_l1_regularizer(weights):
        return tf.reduce_sum(tf.abs(0.01 * weights))

    def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
        return tf.where(weights < 0., tf.zeros_like(weights), weights)

    model = keras.models.load_model(
        "mainPY/custom/my_model_with_many_custom_parts.h5",
        custom_objects={
        "my_l1_regularizer": my_l1_regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
        })

    class MyL1Regularizer(keras.regularizers.Regularizer):
        def __init__(self, factor):
            self.factor = factor
        def __call__(self, weights):
            return tf.reduce_sum(tf.abs(self.factor * weights))
        def get_config(self):
            return {"factor": self.factor}

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1, activation=my_softplus,
                        kernel_regularizer=MyL1Regularizer(0.01),
                        kernel_constraint=my_positive_weights,
                        kernel_initializer=my_glorot_initializer),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))

    model.save("mainPY/custom/my_model_with_many_custom_parts.h5")


def customMetrics():
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
    model.fit(X_train_scaled, y_train, epochs=2)

    # **Warning**: if you use the same function as the loss and a metric, you may be surprised to see different results. This is generally just due to floating point precision errors: even though the mathematical equations are equivalent, the operations are not run in the same order, which can lead to small differences. Moreover, when using sample weights, there's more than just precision errors:
    # * the loss since the start of the epoch is the mean of all batch losses seen so far. Each batch loss is the sum of the weighted instance losses divided by the _batch size_ (not the sum of weights, so the batch loss is _not_ the weighted mean of the losses).
    # * the metric since the start of the epoch is equal to the sum of weighted instance losses divided by sum of all weights seen so far. In other words, it is the weighted mean of all the instance losses. Not the same thing.
    # 
    # If you do the math, you will find that loss = metric * mean of sample weights (plus some floating point precision error).

    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[create_huber(2.0)])

    sample_weight = np.random.rand(len(y_train))
    history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weight)

    history.history["loss"][0], history.history["huber_fn"][0] * sample_weight.mean()

    # ### Streaming metrics
    precision = keras.metrics.Precision()
    precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
    precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])

    print(precision.result())
    print(precision.variables)
    print(precision.reset_states())


def createStreamingMetric():

    class HuberMetric(keras.metrics.Metric):
        def __init__(self, threshold=1.0, **kwargs):
            super().__init__(**kwargs) # handles base args (e.g., dtype)
            self.threshold = threshold
            #self.huber_fn = create_huber(threshold) # TODO: investigate why this fails
            self.total = self.add_weight("total", initializer="zeros")
            self.count = self.add_weight("count", initializer="zeros")
        def huber_fn(self, y_true, y_pred): # workaround
            error = y_true - y_pred
            is_small_error = tf.abs(error) < self.threshold
            squared_loss = tf.square(error) / 2
            linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        def update_state(self, y_true, y_pred, sample_weight=None):
            metric = self.huber_fn(y_true, y_pred)
            self.total.assign_add(tf.reduce_sum(metric))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        def result(self):
            return self.total / self.count
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}

    # Let's check that the `HuberMetric` class works well:

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])

    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

    model.save("mainPY/custom/my_model_with_a_custom_metric.h5")


def createStreamingMetricV2():

    class HuberMetric(keras.metrics.Mean):
        def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
            self.threshold = threshold
            self.huber_fn = create_huber(threshold)
            super().__init__(name=name, dtype=dtype)
        def update_state(self, y_true, y_pred, sample_weight=None):
            metric = self.huber_fn(y_true, y_pred)
            super(HuberMetric, self).update_state(metric, sample_weight)
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}    

        model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
])

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=keras.losses.Huber(2.0), optimizer="nadam", weighted_metrics=[HuberMetric(2.0)])

    sample_weight = np.random.rand(len(y_train))
    history = model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32),
                        epochs=2, sample_weight=sample_weight)

    history.history["loss"][0], history.history["HuberMetric"][0] * sample_weight.mean()

    model.save("mainPY/custom/my_model_with_a_custom_metric_v2.h5")


def customLayers():

    exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
    exponential_layer([-1., 0., 1.])

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1),
        exponential_layer
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=5,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

def customExponentialLayer():

    exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
    exponential_layer([-1., 0., 1.])

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, batch_input_shape):
            self.kernel = self.add_weight(
                name="kernel", shape=[batch_input_shape[-1], self.units],
                initializer="glorot_normal")
            self.bias = self.add_weight(
                name="bias", shape=[self.units], initializer="zeros")
            super().build(batch_input_shape) # must be at the end

        def call(self, X):
            return self.activation(X @ self.kernel + self.bias)

        def compute_output_shape(self, batch_input_shape):
            return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "units": self.units,
                    "activation": keras.activations.serialize(self.activation)}

    # Adding an exponential layer at the output of a regression model can be useful if the values to predict are positive and with very different scales (e.g., 0.001, 10., 10000):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1),
        exponential_layer
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=5,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    

    model = keras.models.Sequential([
    MyDense(30, activation="relu", input_shape=input_shape),
    MyDense(1)
    ])

    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    model.save("mainPY/custom/my_model_with_a_custom_layer.h5")


def customMultiLayer():

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, batch_input_shape):
            self.kernel = self.add_weight(
                name="kernel", shape=[batch_input_shape[-1], self.units],
                initializer="glorot_normal")
            self.bias = self.add_weight(
                name="bias", shape=[self.units], initializer="zeros")
            super().build(batch_input_shape) # must be at the end

        def call(self, X):
            return self.activation(X @ self.kernel + self.bias)

        def compute_output_shape(self, batch_input_shape):
            return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "units": self.units,
                    "activation": keras.activations.serialize(self.activation)}

    model = keras.models.load_model("mainPY/custom/my_model_with_a_custom_layer.h5",
                                custom_objects={"MyDense": MyDense})
                
    class MyMultiLayer(keras.layers.Layer):
        def call(self, X):
            X1, X2 = X
            return X1 + X2, X1 * X2

        def compute_output_shape(self, batch_input_shape):
            batch_input_shape1, batch_input_shape2 = batch_input_shape
            return [batch_input_shape1, batch_input_shape2]


    inputs1 = keras.layers.Input(shape=[2])
    inputs2 = keras.layers.Input(shape=[2])
    outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))

    # Let's create a layer with a different behavior during training and testing:

    class AddGaussianNoise(keras.layers.Layer):
        def __init__(self, stddev, **kwargs):
            super().__init__(**kwargs)
            self.stddev = stddev

        def call(self, X, training=None):
            if training:
                noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
                return X + noise
            else:
                return X

        def compute_output_shape(self, batch_input_shape):
            return batch_input_shape

    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

def customModelsSave():

    X_new_scaled = X_test_scaled

    class ResidualBlock(keras.layers.Layer):
        def __init__(self, n_layers, n_neurons, **kwargs):
            super().__init__(**kwargs)
            self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                            kernel_initializer="he_normal")
                        for _ in range(n_layers)]

        def call(self, inputs):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            return inputs + Z

    class ResidualRegressor(keras.models.Model):
        def __init__(self, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = keras.layers.Dense(30, activation="elu",
                                            kernel_initializer="he_normal")
            self.block1 = ResidualBlock(2, 30)
            self.block2 = ResidualBlock(2, 30)
            self.out = keras.layers.Dense(output_dim)

        def call(self, inputs):
            Z = self.hidden1(inputs)
            for _ in range(1 + 3):
                Z = self.block1(Z)
            Z = self.block2(Z)
            return self.out(Z)

    model = ResidualRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=5)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)

    model.save("mainPY/custom/my_custom_model.ckpt")

def customModelsLoad():

    model = keras.models.load_model("mainPY/custom/my_custom_model.ckpt")
    history = model.fit(X_train_scaled, y_train, epochs=5)

    # We could have defined the model using the sequential API instead:

    X_new_scaled = X_test_scaled

    class ResidualBlock(keras.layers.Layer):
        def __init__(self, n_layers, n_neurons, **kwargs):
            super().__init__(**kwargs)
            self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                            kernel_initializer="he_normal")
                        for _ in range(n_layers)]

        def call(self, inputs):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            return inputs + Z

    block1 = ResidualBlock(2, 30)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
        block1, block1, block1, block1,
        ResidualBlock(2, 30),
        keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=5)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)


def customLosses_Metrics_based_on_Model_Internals():

    X_new_scaled = X_test_scaled

    class ReconstructingRegressor(keras.models.Model):
        def __init__(self, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.hidden = [keras.layers.Dense(30, activation="selu",
                                            kernel_initializer="lecun_normal")
                        for _ in range(5)]
            self.out = keras.layers.Dense(output_dim)
            # TODO: check https://github.com/tensorflow/tensorflow/issues/26260
            #self.reconstruction_mean = keras.metrics.Mean(name="reconstruction_error")

        def build(self, batch_input_shape):
            n_inputs = batch_input_shape[-1]
            self.reconstruct = keras.layers.Dense(n_inputs)
            super().build(batch_input_shape)

        def call(self, inputs, training=None):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            reconstruction = self.reconstruct(Z)
            recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
            self.add_loss(0.05 * recon_loss)
            #if training:
            #    result = self.reconstruction_mean(recon_loss)
            #    self.add_metric(result)
            return self.out(Z)

    model = ReconstructingRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=2)
    y_pred = model.predict(X_test_scaled)

def computing_gradients_using_autodiff():

    l2_reg = keras.regularizers.l2(0.05)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                        kernel_regularizer=l2_reg),
        keras.layers.Dense(1, kernel_regularizer=l2_reg)
    ])

    def random_batch(X, y, batch_size=32):
        idx = np.random.randint(len(X), size=batch_size)
        return X[idx], y[idx]

    def print_status_bar(iteration, total, loss, metrics=None):
        metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                            for m in [loss] + (metrics or [])])
        end = "" if iteration < total else "\n"
        print("\r{}/{} - ".format(iteration, total) + metrics,
            end=end)

    mean_loss = keras.metrics.Mean(name="loss")
    mean_square = keras.metrics.Mean(name="mean_square")
    for i in range(1, 50 + 1):
        loss = 1 / i
        mean_loss(loss)
        mean_square(i ** 2)
        print_status_bar(i, 50, mean_loss, [mean_square])
        time.sleep(0.05)

    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.MeanAbsoluteError()]

    for epoch in range(1, n_epochs + 1):
        print("Epoch {}/{}".format(epoch, n_epochs))
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train_scaled, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            for variable in model.variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))
            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
        print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
        for metric in [mean_loss] + metrics:
            metric.reset_states()


    try:
        from tqdm.notebook import trange
        from collections import OrderedDict
        with trange(1, n_epochs + 1, desc="All epochs") as epochs:
            for epoch in epochs:
                with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs)) as steps:
                    for step in steps:
                        X_batch, y_batch = random_batch(X_train_scaled, y_train)
                        with tf.GradientTape() as tape:
                            y_pred = model(X_batch)
                            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                            loss = tf.add_n([main_loss] + model.losses)
                        gradients = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        for variable in model.variables:
                            if variable.constraint is not None:
                                variable.assign(variable.constraint(variable))                    
                        status = OrderedDict()
                        mean_loss(loss)
                        status["loss"] = mean_loss.result().numpy()
                        for metric in metrics:
                            metric(y_batch, y_pred)
                            status[metric.name] = metric.result().numpy()
                        steps.set_postfix(status)
                for metric in [mean_loss] + metrics:
                    metric.reset_states()
    except ImportError as ex:
        print("To run this cell, please install tqdm, ipywidgets and restart Jupyter")



def my_mse(y_true, y_pred):
    print("Tracing loss my_mse()")
    return tf.reduce_mean(tf.square(y_pred - y_true))

def my_mae(y_true, y_pred):
    print("Tracing metric my_mae()")
    return tf.reduce_mean(tf.abs(y_pred - y_true))

def customLayerModel():

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel', 
                                        shape=(input_shape[1], self.units),
                                        initializer='uniform',
                                        trainable=True)
            self.biases = self.add_weight(name='bias', 
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
            super().build(input_shape)

        def call(self, X):
            print("Tracing MyDense.call()")
            return self.activation(X @ self.kernel + self.biases)

    class MyModel(keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = MyDense(30, activation="relu")
            self.hidden2 = MyDense(30, activation="relu")
            self.output_ = MyDense(1)

        def call(self, input):
            print("Tracing MyModel.call()")
            hidden1 = self.hidden1(input)
            hidden2 = self.hidden2(hidden1)
            concat = keras.layers.concatenate([input, hidden2])
            output = self.output_(concat)
            return output

    model = MyModel()

    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)


def customLayerModelDynamic():

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel', 
                                        shape=(input_shape[1], self.units),
                                        initializer='uniform',
                                        trainable=True)
            self.biases = self.add_weight(name='bias', 
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
            super().build(input_shape)

        def call(self, X):
            print("Tracing MyDense.call()")
            return self.activation(X @ self.kernel + self.biases)

    class MyModel(keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = MyDense(30, activation="relu")
            self.hidden2 = MyDense(30, activation="relu")
            self.output_ = MyDense(1)

        def call(self, input):
            print("Tracing MyModel.call()")
            hidden1 = self.hidden1(input)
            hidden2 = self.hidden2(hidden1)
            concat = keras.layers.concatenate([input, hidden2])
            output = self.output_(concat)
            return output
    # You can turn this off by creating the model with `dynamic=True` (or calling `super().__init__(dynamic=True, **kwargs)` in the model's constructor):

    model = MyModel(dynamic=True)

    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae])

    # Not the custom code will be called at each iteration. Let's fit, validate and evaluate with tiny datasets to avoid getting too much output:
    model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
            validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
    model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)

def customLayerModelRun_eagerly():

    class MyDense(keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = keras.activations.get(activation)

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel', 
                                        shape=(input_shape[1], self.units),
                                        initializer='uniform',
                                        trainable=True)
            self.biases = self.add_weight(name='bias', 
                                        shape=(self.units,),
                                        initializer='zeros',
                                        trainable=True)
            super().build(input_shape)

        def call(self, X):
            print("Tracing MyDense.call()")
            return self.activation(X @ self.kernel + self.biases)

    class MyModel(keras.models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden1 = MyDense(30, activation="relu")
            self.hidden2 = MyDense(30, activation="relu")
            self.output_ = MyDense(1)

        def call(self, input):
            print("Tracing MyModel.call()")
            hidden1 = self.hidden1(input)
            hidden2 = self.hidden2(hidden1)
            concat = keras.layers.concatenate([input, hidden2])
            output = self.output_(concat)
            return output

    model = MyModel()
    model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae], run_eagerly=True)

    model.fit(X_train_scaled[:64], y_train[:64], epochs=1,
            validation_data=(X_valid_scaled[:64], y_valid[:64]), verbose=0)
    model.evaluate(X_test_scaled[:64], y_test[:64], verbose=0)
# Custom Loss
# customLossSaveWeights()
# customLossLoadWeights()
# customLossSaveThreshold_2()
# customLossLoadThreshold_2()
# customLossWithClassSave()
# customLossWithClassLoad() # Error

# Custom Functions
# customFunctionsSave()
# customFunctionsClassL1Regularization()

# Custom Metrics
# customMetrics()
# createStreamingMetric()
# createStreamingMetricV2()

# Custom Layers
# customLayers()
# customExponentialLayer()
# customMultiLayer()

# Custom Models
# customModelsSave()
# customModelsLoad()

# customLosses_Metrics_based_on_Model_Internals()

# computing_gradients_using_autodiff()

# customLayerModel()
# customLayerModelDynamic()
customLayerModelRun_eagerly()