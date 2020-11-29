import tensorflow as tf

from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

print(diabetes_dataset.keys())

data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

targets = (targets - targets.mean(axis=0)) / targets.std()

from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_model():

    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1))

    return model

model = get_model()


model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(train_data, train_targets, epochs=1000,
            validation_split=0.15, batch_size=64, verbose=False)

model.evaluate(test_data, test_targets, verbose=2)

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss vs Epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="Upper right")
plt.show()