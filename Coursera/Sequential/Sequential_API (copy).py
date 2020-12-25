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


# %%

# Early Stopping and Patience

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([

    Conv1D(16,5 , activation="relu", input_shape=(128,1)),
    MaxPool1D(4),
    Flatten(),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stoppingg = EarlyStopping(monitor="val_acuracy", patience=5, min_delta=0.01, mode="max")

model.fit(X_train, y_train, validation_split=0.2, epochs=100,
    callbacks=[early_stoppingg])


# %%

# Tutorial Early Stopping/Patience



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

# %%

# Saving and loading model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(64, activation="sigmoid", input_shape=(10,)),
    Dense(1)
])
model.compile(optimizer="sgd", loss=BinaryCrossentropy(from_logits=True))


# Save only weight of the model using tensorflow
# Tensorflow saves in the checkpoint format for each epoch
checkpoint = ModelCheckpoint("my_model", save_weights_only=True)
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])
# checkpoint
# my_model.data-0000-of-0001
# my_model.index


# save only weights of the model using Keras 
# Keras uses HDF5 .h5 file fomat for saving the weights
checkpoint = ModelCheckpoint("keras_model.h5", save_weights_only=True)
model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])


# Load the trained weights of the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(64, activation="sigmoid", input_shape=(10,)),
    Dense(1)
])

model.load_weights("keras_model.hs")



# Save model weights manually
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation="sigmoid", input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
early_stoppingg = EarlyStopping(monitor="val_mae", patience=3)
model.fit(X_train, y_train, validation_split=0.2, epochs=10)

model.save_weights("my_model")

# %%
# Tutorial Saving and loading model weights

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
y_train = y_train/255.0

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

fig, ax =  plt.subplots(1, 10, figsize=(10,1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])


def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=1)
    print("accuracy: {acc:0.3f}".format(acc=test_acc))



def get_new_model():
    model = Sequential([

        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3), activation="relu", name = "Conv_1"),
        Conv2D(filters=9, kernel_size=(3,3), activation="relu", name = "Conv_2"),
        MaxPool2D(pool_size=(4,4), name="Pool_1"),
        Flatten(name = "Flatten"),
        Dense(units=32, activation="relu", name="dense_1"),
        Dense(units=10, activation="softmax", name= "dense_2")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

model = get_new_model()
print(model.summary())

checkpoint_path = "Coursera/Sequential_model_checkpoints/checkpoint"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, frequency="epoch", save_weights_only=True, verbose=1)

model.fit(x=x_train, y=y_train, epochs=3, callbacks=[checkpoint])


model = get_new_model()
get_test_accuracy(model, x_test, y_test)

model.load_weights(checkpoint_path)
val_acc = get_test_accuracy(model, x_test, y_test)

print(val_acc)




# %%

# Model Saving Criteria

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([

    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", 
            metrics=["acc", "mae"])
    
checkpoint = ModelCheckpoint("training_run_1/my_model", save_weights_only=True,
                            save_best_only=True, monitor="val_acc", mode="max")


checkpoint = ModelCheckpoint("training_run_1/my_model.{epoch}.{batch}", 
        save_weights_only=True, save_freq=1000)

checkpoint = ModelCheckpoint("training_run_1/my_model.{epoch}-{val_loss:.4f}", 
        save_weights_only=True)


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10,
        batch_size=16, callbacks=[checkpoint])

# %%

# Tutorial Model Saving Criteria
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
y_train = y_train/255.0

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

fig, ax =  plt.subplots(1, 10, figsize=(10,1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])


def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=1)
    print("accuracy: {acc:0.3f}".format(acc=test_acc))



def get_new_model():
    model = Sequential([

        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3), activation="relu", name = "Conv_1"),
        Conv2D(filters=9, kernel_size=(3,3), activation="relu", name = "Conv_2"),
        MaxPool2D(pool_size=(4,4), name="Pool_1"),
        Flatten(name = "Flatten"),
        Dense(units=32, activation="relu", name="dense_1"),
        Dense(units=10, activation="softmax", name= "dense_2")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

checkpoint_5000_path = "model_checkpoints_5000/checkpoint_{epoch:02d}_{batch:04d}"
checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path,
                        save_weights_only=True,
                        save_freq=5000,
                        verbose=1)


model = get_new_model()
model.fit(x=x_train, y=y_train, epochs=3, validation_data=(x_test, y_test),
            batch_size=10, callbacks=[checkpoint_5000])



x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]


model = get_new_model()

checkpoint_best_path = "model_checkpoints_best/checkpoint"
checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path,
        save_weights_only=True,
            save_freq="epoch",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1)

history = model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test),
        batch_size=10, callbacks=[checkpoint_best],
            verbose=0)

import pandas as pd

df = pd.DataFrame(history.history)
df.plot(y=["accuracy", "val_accuracy"])


new_model = get_new_model()
new_model.load_weights(checkpoint_best_path)
get_test_accuracy(new_model, x_test, y_test)


# %%

# Saving entire model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax"),
])

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
        metrics=["acc", "mae"])

checkpoint = ModelCheckpoint("my_model", save_weights_only=False)
# my_model/assets/
# my_model/saved_model.pb
# my_model/variables/variables.data-0000-of-0001
# my_model/variables/variable.index

checkpoint = ModelCheckpoint("keras_model.h5", save_weights_only=False)
# keras_model.h5

model.save("my_model")
# Manually save entire model # SavedModel format

model.save("keras_model.h5")
# HDF5 format keras

model.fit(x_train,y_train, epochs=10, callbacks=[checkpoint])




from tensorflow.keras.model import load_model

# Load tensorflow model
new_model = load_model("my_model")

# Load keras model
new_keras_model = load_model("keras_model.h5")

new_model.summary()
new_model.fit(x_train, y_train, validation_data(X_val, y_val),
            epochs=20, batch_size=16)
new_model.evaluate(x_test, y_test)
new_model.predict(x_sample)


# %%


# Tutorial Saving the entire model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
y_train = y_train/255.0

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

fig, ax =  plt.subplots(1, 10, figsize=(10,1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])


def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=1)
    print("accuracy: {acc:0.3f}".format(acc=test_acc))



def get_new_model():
    model = Sequential([

        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3), activation="relu", name = "Conv_1"),
        Conv2D(filters=9, kernel_size=(3,3), activation="relu", name = "Conv_2"),
        MaxPool2D(pool_size=(4,4), name="Pool_1"),
        Flatten(name = "Flatten"),
        Dense(units=32, activation="relu", name="dense_1"),
        Dense(units=10, activation="softmax", name= "dense_2")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


checkpoint_path = "model_checkpoints"
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
            save_weights_only=False,
                frequency="epoch",
                    verbose=1)
        
model = get_new_model()
model.fit(x=x_train, y=y_train, epochs=3,
            callbacks=[checkpoint])
            
print(get_test_accuracy(model, x_test, y_test))

from tensorflow.keras.models import load_model

model = load_model(checkpoint_path)
print(get_test_accuracy(model, x_test, y_test))


# Use the .h5 format to save model

model.save("my_model.h5")


model = load_model("my_model.h5")
print(get_test_accuracy(model, x_test, y_test))

# %%

# Loading pre-trained keras models

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights="imagenet", include_top=True)

img_input = image.load_img("my_picture.jpg", target_size=(224,224))
img_input = image.img_to_array(img_input)
img_input = preprocess_input(img_input[np.newaxis, ...])

preds = model.predict(img_input)
decode_predictions = decode_predictions(preds, top=3)[0]




# %%

# Tutorial pre-trained keras models

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights="imagenet")

print(model.summary())


from tensorflow.keras.preprocessing.image import load_img

lemon_img = load_img("data/lemon.jpg", target_size=(224, 224))
viaduct_img = load_img("data/viaduct.jpg", target_size=(224, 224))
water_img = load_img("data/water_tower.jpg", target_size=(224, 224))

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd

def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis,...]
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x), top=5)
    top_preds = pd.DataFrame(columns=["prediction", "probability"],
                    index= np.arrange(5)+1)


    for i in range(5):
        top_preds.loc[i+1, "prediction"] = preds[0][i][1]
        top_preds.loc[i+1, "probability"] = preds[0][i][2]

    return top_preds


get_top_5_predictions(water_img)


# %%
