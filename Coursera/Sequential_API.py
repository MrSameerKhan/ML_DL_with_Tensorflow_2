# ANN

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

print(model.summary())


#%%
#Tutorial ANN

from tensorflow.keras.layers import Flatten, Softmax, Dense
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation="softmax"))


print(model.summary())

# %%
# CNN

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential


model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding="same", activation="relu", input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=3))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

print(model.summary())



# %%

# Tutorial CNN

from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(16,(3,3), activation="relu", input_shape=(1,28,28), data_format="channels_first"))
model.add(MaxPool2D(pool_size=(3,3), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))


print(model.summary())

# %%
# Compile ANN
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation="elu", input_shape=(32,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss= "binary_crossentropy", metrics=["accuracy", "mse"])

print(model.summary())


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.6), tf.keras.metrics.MeanAbsoluteError()])

print(model.summary())

# %%

# Tutorial Compile ANN

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation="elu", input_shape=(32,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print(model.optimizer)
print(model.loss)
print(model.metrics)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(opt,loss="categorical_crossentropy",
            metrics=["accuracy", "mse"])


print(model.optimizer)
print(model.loss)
print(model.metrics)
print(model.optimizer.lr)

# %%
# Fit ANN




import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64,input_shape=(32,)))
model.add(Dense(1, activation="sigmoid"))

opt = tf.keras.optimizers.Adam()
sce = tf.keras.losses.SparseCategoricalCrossentropy()
acc = tf.keras.metrics.BinaryAccuracy()
mse = tf.keras.metrics.MeanAbsoluteError()

model.compile(opt, sce, mse)

print(model.summary())

# history = model.fit(x_train, y_train, batch_size=2, epochs=20)



# %%

# Tutorial Fit

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mse = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=[acc, mse])

print(model.summary())

(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

history = model.fit(train_images[...,np.newaxis],train_labels, batch_size=256, epochs=8)

df = pd.DataFrame(history.history)
print(df.head())


loss_plot = df.plot(y="loss", title= "Loss vs Epochs", legend = True)
loss_plot.set(xlabel="Epochs", ylabel="Loss")


# %%
