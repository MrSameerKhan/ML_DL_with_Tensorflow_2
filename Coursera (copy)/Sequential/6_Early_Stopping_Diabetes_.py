import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


diabetes_dataset = load_diabetes()

data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]

targets = (targets - targets.mean(axis=0)) / targets.std()

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)


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

unregularized_model = get_model()
unregularized_model.compile(optimizer="adam",loss="mae")
unreg_history = unregularized_model.fit(train_data, train_targets, epochs=100, 
        validation_split=0.15, batch_size=64, verbose=False, callbacks=[tf.keras.callbacks.EarlyStopping()])


unregularized_model.evaluate(test_data, test_targets, verbose=2)


regularized_model = get_regularized_model(1e-5, 0.2)
regularized_model.compile(optimizer="adam", loss="mse")
reg_history = regularized_model.fit(train_data, train_targets, epochs=100,
        validation_split=0.15, batch_size=64, verbose=False,
        callbacks=[tf.keras.callbacks.EarlyStopping()])
regularized_model.evaluate(test_data, test_targets, verbose=2)


fig = plt.figure(figsize=(12,5))
fig.add_subplot(121)

plt.plot(unreg_history.history["loss"])
plt.plot(unreg_history.history["val_loss"])
plt.title("Unregularised model: loss vs epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")

fig.add_subplot(122)

plt.plot(reg_history.history["loss"])
plt.plot(reg_history.history["val_loss"])
plt.title("Unregularised model: loss vs epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc="upper right")


plt.show()