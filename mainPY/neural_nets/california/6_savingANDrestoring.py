
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

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# # Saving and Restoring
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]

model.save("./mainPY/neural_nets/california/6_save.h5")
model_h5 = keras.models.load_model("./mainPY/neural_nets/california/6_save.h5")
y_pred_h5 = model_h5.predict(X_new)
print("Saving and Restoring h5 : ", y_pred_h5)

model.save_weights("./mainPY/neural_nets/california/6_save_weights.ckpt")
model_ckpt = model.load_weights("./mainPY/neural_nets/california/6_save_weights.ckpt")
# y_pred_ckpt = model_ckpt.predict(X_new)
# print(y_pred_ckpt)  # AttributeError: 'CheckpointLoadStatus' object has no attribute 'predict'