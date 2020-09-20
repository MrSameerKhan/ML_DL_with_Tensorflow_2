
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# X_train(60000,28,28) y_train(60000,1)
# X_test(10000, 28, 28) y_test (10000,1)

# normalization of pixels
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

print(X_valid.shape, X_train.shape, y_valid.shape, y_train.shape, X_test.shape)
# X_valid.shape,   X_train.shape,   y_valid.shape,  y_train.shape,  X_test.shape
# (5000, 28, 28)  (55000, 28, 28)   (5000,)         (55000,)        (10000, 28, 28)
# there are 10 no. of classes represented from 0 to 9


import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap="binary")
plt.axis("off")
#plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# y_train[0] = 4 , class_names = coat
print(class_names[y_train[0]])


# model architecture
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# sabe model architecture with parameters
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)

# configure the model
model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="sgd", metrics=["accuracy"])

# train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
print(y_pred)

np.array(class_names)[y_pred]

y_new = y_test[:3]
print(y_new)