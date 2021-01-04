
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
print(df)


testLoss, testAcc, testMSE = model.evaluate(test_images[...,np.newaxis], test_labels, verbose=2)

print(testLoss, testAcc, testMSE)


loss_plot = df.plot(y="loss", title= "Loss vs Epochs", legend = True)
loss_plot.set(xlabel="Epochs", ylabel="Loss")


