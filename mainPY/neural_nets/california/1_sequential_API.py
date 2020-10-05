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

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

# plt.plot(pd.DataFrame(history.history))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()

print("Sequential API  : ", y_pred)


"""
The predict_classes method is only available for the Sequential class 
(which is the class of your first model) but not for the Model class
 (the class of your second model).
With the Model class, you can use the predict method 
which will give you a vector of probabilities and 
then get the argmax of this vector (with np.argmax(y_pred1,axis=1)).
"""