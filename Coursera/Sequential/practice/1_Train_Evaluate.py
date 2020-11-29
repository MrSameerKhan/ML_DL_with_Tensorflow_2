
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

model = Sequential([

    Conv2D(filters=16, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    MaxPool2D(pool_size=(3,3)),
    Conv2D(filters=16, kernel_size=(3,3), activation="relu"),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")

])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
los = tf.keras.losses.SparseCategoricalCrossentropy()
mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt, loss=los, metrics=["acc", mae])

history = model.fit(train_images, train_labels, epochs=10, batch_size=256)