
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes_dataset = load_diabetes()

data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

targets = (targets - targets.mean(axis=0)) / targets.std()

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

def get_regularized_model(wd, rate):
    model=Sequential([

        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(1)

    ])

    return model


model = get_regularized_model(1e-5, 0.3)

model.compile(optimizer="adam", loss="mae", metrics=["mae"])

history = model.fit(train_data, train_targets, epochs=100, validation_split=0.15,
        batch_size=64, verbose=False)

model.evaluate(test_data, test_targets, verbose=2)



import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss vs Epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="Upper right")
plt.show()
