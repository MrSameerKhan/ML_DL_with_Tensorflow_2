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

# Evaluate


import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(1, activation="sigmoid", input_shape=(12,)))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train)

loss, accuracy , mae = model.evaluate(x_test, y_test)

pred = model.predict(x_sample)

# %%

# Tutorial Evaluate

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]



model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation="relu", 
    input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(16,activation="relu", kernel_size=3))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))

print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mse = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=[acc, mse])


# history = model.fit(train_images[...,np.newaxis], train_labels, batch_size= 256, epochs=32)
history = model.fit(train_images[...,np.newaxis],train_labels, batch_size=256, epochs=8)


df = pd.DataFrame(history.history)
print(df.head())
print(df.tail())

testLoss, testAcc, testMSE = model.evaluate(test_images[...,np.newaxis], test_labels, verbose=2)

print(testLoss, testAcc, testMSE)


loss_plot = df.plot(y="loss", title= "Loss vs Epochs", legend = True)
loss_plot.set(xlabel="Epochs", ylabel="Loss")


random_inx = np.random.choice(test_images.shape[0])
inx = 30

test_image = test_images[inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[inx]]}")

predictions = model.predict(test_image[np.newaxis,...,np.newaxis])
print(f"Model prediction: {labels[np.argmax(predictions)]}")

# %%
# Validation Sets

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dense(2))
opt = Adam(learning_rate=0.05)
model.compile(optimizer=opt, loss="mse", metrics=["mape"])

history = model.fit(inputs, targets, validation_split=0.2)

print(history.history.keys())



import tensorflow as tf

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

model.fit(X_train, y_train, validation_data=(X_test, y_test))


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

model.fit(X_train, y_train, validation_data=(X_val, y_val))


# %%

# Tutorial validation Sets

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

history = model.fit(train_data, train_targets, epochs=100,
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

# %%

# Model Regularization

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005, l2=0.001),
        bias_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dense(1, activation="sigmoid"))

# model.add(Dense(64, activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l1(0.005)))

# model.add(Dense(64, activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2=0.001))

model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["acc"])
model.fit(inputs, targets,validation_split=0.25)



model = Sequential()
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))


model.compile(optimizer="adadelta", loss="binary_crossentropy",metrics=["acc"])  # Training with Dropou
model.fit(inputs, targets, validation_split=0.25) # Testing, no dropout
model.predict(test_inputs)# Testing, no dropout

# %%
# Tutorial Model Regularization

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



# %%
# Callbacks


class my_callback(callback):

    def on_train_begin(self, logs=None):
        pass
    def on_train_batch_begin(self, batch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        pass

history= model.fit(xtrain, ytrain, epochs=5, callbacks=[my_callbacks])

# %%

# Tutorial Callbacks


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import Callback

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


class TrainingCallback(Callback):

    def on_train_begin(self, logs=None):
        print("Starting training .... ")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training Starting bacth {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
    
    def on_train_end(self, logs=None):
        print("Finished training")


class TestingCallback(Callback):

    def on_test_begin(self, logs=None):
        print("Starting testing .... ")

    def on_test_batch_begin(self, batch, logs=None):
        print(f"Testing Starting bacth {batch}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"Testing: Finished batch {batch}")

    def on_testing_end(self, logs=None):
        print("Finished testing")


class PredictionCallback(Callback):

    def on_predict_begin(self, logs=None):
        print("Starting prediction .... ")

    def on_predict_batch_begin(self, batch, logs=None):
        print(f"Prediction: Starting bacth {batch}")

    def on_predict_batch_end(self, batch, logs=None):
        print(f"Prediction: Finished batch {batch}")

    def on_predicting_end(self, logs=None):
        print("Finished Prediction")

model = get_regularized_model(1e-5, 0.3)

model.compile(optimizer="adam", loss="mse")

model.fit(train_data, train_targets, epochs=3, batch_size=128, verbose=False, callbacks=[TrainingCallback()])

model.evaluate(test_data, test_targets, verbose=False, callbacks=[TestingCallback()])

model.predict(test_data, verbose=False, callbacks=[PredictionCallback()])


