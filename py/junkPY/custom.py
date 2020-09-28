# Tensors and Operations
import tensorflow as tf
import numpy as np


# Custom Loss Function
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow import keras

class tensorsANDoperations:

    print("Begin Tensor")

    def tensorsOperations(self):

        # matrix
        t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) 

        # scalar
        scalar = tf.constant(42) 

        t.shape
        t.dtype

        # indexing
        t[:, 1:]

        t[..., 1, tf.newaxis]

        # ops
        t + 10

        tf.square(t)

        t @ tf.transpose(t)

        # Using keras.backend
        from tensorflow import keras
        K = keras.backend
        K.square(K.transpose(t)) + 10

        # From/To Numpy
        a = np.array([2., 4., 5.])
        tf.constant(a)

        t.numpy()

        np.array(t)

        tf.square(a)

        np.square(t)

        # conflicting types
        try:
            tf.constant(2.0) + tf.constant(40)
        except tf.errors.InvalidArgumentError as ex:
            print(ex)

        try:
            tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
        except tf.errors.InvalidArgumentError as ex:
            print(ex)

        #Strings
        tf.constant(b"hello world")

        tf.constant("café")

        u = tf.constant([ord(c) for c in "café"])

        b = tf.strings.unicode_encode(u, "UTF-8")
        tf.strings.length(b, unit="UTF8_CHAR")

        tf.strings.unicode_decode(b, "UTF-8")

        # String Arrays
        p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])

        tf.strings.length(p, unit="UTF8_CHAR")

        r = tf.strings.unicode_decode(p, "UTF8")

        #Ragged tensors 
        print(r[1])
        print(r[1:3])

        r2 = tf.ragged.constant([[65, 66], [], [67]])
        print(tf.concat([r, r2], axis=0))

        r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
        print(tf.concat([r, r3], axis=1))

        tf.strings.unicode_encode(r3, "UTF-8")

        r.to_tensor()

        #Sparse tensors
        s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])

        tf.sparse.to_dense(s)

        s2 = s * 2.0

        try:
            s3 = s + 1.
        except TypeError as ex:
            print(ex)

        s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
        tf.sparse.sparse_dense_matmul(s, s4)

        s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],
                        values=[1., 2.],
                        dense_shape=[3, 4])
        print(s5)

        try:
            tf.sparse.to_dense(s5)
        except tf.errors.InvalidArgumentError as ex:
            print(ex)

        s6 = tf.sparse.reorder(s5)
        tf.sparse.to_dense(s6)

        #Sets
        set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
        set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
        tf.sparse.to_dense(tf.sets.union(set1, set2))

        tf.sparse.to_dense(tf.sets.difference(set1, set2))

        tf.sparse.to_dense(tf.sets.intersection(set1, set2))

        #Variables
        v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

        v.assign(2 * v)

        v[0, 1].assign(42)

        v[:, 2].assign([0., 1.])

        try:
            v[1] = [7., 8., 9.]
        except TypeError as ex:
            print(ex)

        v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.])

        sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                        indices=[1, 0])
        v.scatter_update(sparse_delta)


        # Tensor Arrays
        array = tf.TensorArray(dtype=tf.float32, size=3)
        array = array.write(0, tf.constant([1., 2.]))
        array = array.write(1, tf.constant([3., 10.]))
        array = array.write(2, tf.constant([5., 7.]))

        array.read(1)

        array.stack()

        mean, variance = tf.nn.moments(array.stack(), axes=0)

        print("End Tensor")



class customLossFunction:

    print("Begin Loss Function")

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)


    def huberFunctionTrain(self):

        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < 1
            squared_loss = tf.square(error) / 2
            linear_loss  = tf.abs(error) - 0.5
            return tf.where(is_small_error, squared_loss, linear_loss)

        plt.figure(figsize=(8, 3.5))
        z = np.linspace(-4, 4, 200)
        plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
        plt.plot(z, z**2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
        plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
        plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
        plt.gca().axhline(y=0, color='k')
        plt.gca().axvline(x=0, color='k')
        plt.axis([-4, 4, 0, 4])
        plt.grid(True)
        plt.xlabel("$z$")
        plt.legend(fontsize=14)
        plt.title("Huber loss", fontsize=14)
        plt.show()


        input_shape = self.X_train.shape[1:]

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                            input_shape=input_shape),
            keras.layers.Dense(1),
        ])

        model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
            validation_data=(self.X_valid_scaled, self.y_valid))

        model.save("my_model_with_a_custom_loss.h5")


    def huberFunctionLoad(self):

        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < 1
            squared_loss = tf.square(error) / 2
            linear_loss  = tf.abs(error) - 0.5
            return tf.where(is_small_error, squared_loss, linear_loss)

        model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
            validation_data=(self.X_valid_scaled, self.y_valid))
    

    def createHuberTrain(self):

        def create_huber(threshold=1.0):
            def huber_fn(y_true, y_pred):
                error = y_true - y_pred
                is_small_error = tf.abs(error) < threshold
                squared_loss = tf.square(error) / 2
                linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
                return tf.where(is_small_error, squared_loss, linear_loss)
            return huber_fn

        input_shape = self.X_train.shape[1:]

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                            input_shape=input_shape),
            keras.layers.Dense(1),
        ])

        model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
            validation_data=(self.X_valid_scaled, self.y_valid))

        model.save("my_model_with_a_custom_loss_threshold_2.h5")

    def createHuberLoad(self):

        def create_huber(threshold=1.0):
            def huber_fn(y_true, y_pred):
                error = y_true - y_pred
                is_small_error = tf.abs(error) < threshold
                squared_loss = tf.square(error) / 2
                linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
                return tf.where(is_small_error, squared_loss, linear_loss)
            return huber_fn

        model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
            validation_data=(self.X_valid_scaled, self.y_valid))
    

    # Custom loss with class 
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

    input_shape = X_train.shape[1:]

    model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))

    model.save("my_model_with_a_custom_loss_class.h5")

    print("End Loss Function")

class otherCustomFunctions:
    print("Begin Custom Function")

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    def my_softplus(self,z): # return value is just tf.nn.softplus(z)
        return tf.math.log(tf.exp(z) + 1.0)

    def my_glorot_initializer(self, shape, dtype=tf.float32):
        stddev = tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)

    def my_l1_regularizer(self,weights):
        return tf.reduce_sum(tf.abs(0.01 * weights))

    def my_positive_weights(self,weights): # return value is just tf.nn.relu(weights)
        return tf.where(weights < 0., tf.zeros_like(weights), weights)



    def train_my_l1_regularizer(self):

        layer = keras.layers.Dense(1, activation=self.my_softplus,
                        kernel_initializer=self.my_glorot_initializer,
                        kernel_regularizer=self.my_l1_regularizer,
                        kernel_constraint=self.my_positive_weights)
    
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        input_shape = self.X_train.shape[1:]

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                            input_shape=input_shape),
            keras.layers.Dense(1, activation=self.my_softplus,
                            kernel_regularizer=self.my_l1_regularizer,
                            kernel_constraint=self.my_positive_weights,
                            kernel_initializer=self.my_glorot_initializer),
        ])

        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
                validation_data=(self.X_valid_scaled, self.y_valid))

        model.save("my_model_with_many_custom_parts.h5")
        

    def trainMyL1Regularizer(self):
        
        model = keras.models.load_model(
            "my_model_with_many_custom_parts.h5",
            custom_objects={"my_l1_regularizer": self.my_l1_regularizer,
            "my_positive_weights": self.my_positive_weights,
            "my_glorot_initializer": self.my_glorot_initializer,
            "my_softplus": self.my_softplus,
            })

        class MyL1Regularizer(keras.regularizers.Regularizer):
            def __init__(self, factor):
                self.factor = factor
            def __call__(self, weights):
                return tf.reduce_sum(tf.abs(self.factor * weights))
            def get_config(self):
                return {"factor": self.factor}


        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42) 

        input_shape = self.X_train.shape[1:]   

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                            input_shape=input_shape),
            keras.layers.Dense(1, activation=self.my_softplus,
                            kernel_regularizer=MyL1Regularizer(0.01),
                            kernel_constraint=self.my_positive_weights,
                            kernel_initializer=self.my_glorot_initializer),
        ])

        model.compile(loss="mse", optimizer="nadam", metrics=["mae"])

        model.fit(self.X_train_scaled, self.y_train, epochs=2,
                validation_data=(self.X_valid_scaled, self.y_valid))

        model.save("my_model_with_many_custom_parts.h5")

        model = keras.models.load_model(
            "my_model_with_many_custom_parts.h5",
            custom_objects={
            "MyL1Regularizer": MyL1Regularizer,
            "my_positive_weights": self.my_positive_weights,
            "my_glorot_initializer": self.my_glorot_initializer,
            "my_softplus": self.my_softplus,
            })
        print("End Custom Function")


class customMetrics:

    print("Begin Custom Metric")

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    def create_huber(self,threshold=1.0):
        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < threshold
            squared_loss = tf.square(error) / 2
            linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        return huber_fn

    input_shape = X_train.shape[1:]

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])

    model.fit(X_train_scaled, y_train, epochs=2)

    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[create_huber(2.0)])

    sample_weight = np.random.rand(len(y_train))
    history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weight)

    history.history["loss"][0], history.history["huber_fn"][0] * sample_weight.mean()

    print("End Custom Metric")
    
"""
class streamingMetrics:

    print("Begin Streaming Metrics")

    # Creating a Streaming Metric

    precision = keras.metrics.Precision()
    precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])

    precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])

    precision.result()

    precision.variables

    precision.reset_states()

    def create_huber(self,threshold=1.0):
        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < threshold
            squared_loss = tf.square(error) / 2
            linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        return huber_fn

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

    m = HuberMetric(2.)

    # total = 2 * |10 - 2| - 2²/2 = 14
    # count = 1
    # result = 14 / 1 = 14
    m(tf.constant([[2.]]), tf.constant([[10.]])) 

    # total = total + (|1 - 0|² / 2) + (2 * |9.25 - 5| - 2² / 2) = 14 + 7 = 21
    # count = count + 2 = 3
    # result = total / count = 21 / 3 = 7
    m(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))

    m.result()

    m.variables

    m.reset_states()
    m.variables

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    input_shape = X_train.shape[1:]

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])

    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

    model.save("my_model_with_a_custom_metric.h5")


    # Streaming Metrics using class
    class HuberMetric(keras.metrics.Mean):
        def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
            self.threshold = threshold
            self.huber_fn = self.create_huber(threshold)
            super().__init__(name=name, dtype=dtype)
        def update_state(self, y_true, y_pred, sample_weight=None):
            metric = self.huber_fn(y_true, y_pred)
            super(HuberMetric, self).update_state(metric, sample_weight)
        def get_config(self):
            base_config = super().get_config()
            return {**base_config, "threshold": self.threshold}        


    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

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

    model.save("my_model_with_a_custom_metric_v2.h5")

    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

    model.metrics[-1].threshold
    print("End Streaming Metrics")
"""

class customLayers:

    print("Begin Custom Layers")

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    input_shape = X_train.shape[1:]

    exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
    exponential_layer([-1., 0., 1.])

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1),
        exponential_layer
    ])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=5,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)


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

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        MyDense(30, activation="relu", input_shape=input_shape),
        MyDense(1)
    ])

    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=2,
            validation_data=(X_valid_scaled, y_valid))
    model.evaluate(X_test_scaled, y_test)

    model.save("my_model_with_a_custom_layer.h5")

    model = keras.models.load_model("my_model_with_a_custom_layer.h5",
                                custom_objects={"MyDense": MyDense})

    class MyMultiLayer(keras.layers.Layer):
        def call(self, X):
            X1, X2 = X
            return X1 + X2, X1 * X2

        def compute_output_shape(self, batch_input_shape):
            batch_input_shape1, batch_input_shape2 = batch_input_shape
            return [batch_input_shape1, batch_input_shape2]

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    inputs1 = keras.layers.Input(shape=[2])
    inputs2 = keras.layers.Input(shape=[2])
    outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))

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

    print("End Custom Layers")






if __name__ == "__main__":
    # tensorflowBasics = tensorsANDoperations()
    # tensorflowBasics.tensorsOperations()

    # CLF = customLossFunction()
    # CLF.huberFunctionTrain()
    # CLF.huberFunctionLoad()
    # CLF.createHuberTrain()
    # CLF.createHuberLoad()

    # OCF = otherCustomFunctions()
    # OCF.train_my_l1_regularizer()
    # OCF.trainMyL1Regularizer() # Error

    # CMetric = customMetrics()
    # SMetric = streamingMetrics() # Error

    # CLayer = customLayers()
    


    print("Begin Custom Models")

    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    input_shape = X_train.shape[1:]

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

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = ResidualRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=5)
    score = model.evaluate(X_test_scaled, y_test)
    y_pred = model.predict(X_new_scaled)

    model.save("my_custom_model.ckpt")

    model = keras.models.load_model("my_custom_model.ckpt")

    history = model.fit(X_train_scaled, y_train, epochs=5)



    #  We could have defined the model using the sequential API instead:

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

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

    print("End Custom Models")



    print("Begin Losses and Metrics based on model internals")
## Losses and Metrics Based on Model Internals
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

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = ReconstructingRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=2)
    y_pred = model.predict(X_test_scaled)

    print("End Losses and Metrics based on model internals")


