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



class customLossFunction():

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
        
    print("Sameer Khan")

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

class otherCustomFunctions():

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



    layer = keras.layers.Dense(1, activation=my_softplus,
                            kernel_initializer=my_glorot_initializer,
                            kernel_regularizer=my_l1_regularizer,
                            kernel_constraint=my_positive_weights)
    
    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    input_shape = X_train.shape[1:]

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

    model.save("my_model_with_many_custom_parts.h5")


    model = keras.models.load_model(
        "my_model_with_many_custom_parts.h5",
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


    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)    

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

    model.save("my_model_with_many_custom_parts.h5")

    model = keras.models.load_model(
        "my_model_with_many_custom_parts.h5",
        custom_objects={
        "MyL1Regularizer": MyL1Regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
        })


# tensorflowBasics = tensorsANDoperations()
# tensorflowBasics.tensorsOperations()

# CLF = customLossFunction()
# CLF.huberFunctionTrain()
# CLF.huberFunctionLoad()
# CLF.createHuberTrain()
# CLF.createHuberLoad()

OCF = otherCustomFunctions()
