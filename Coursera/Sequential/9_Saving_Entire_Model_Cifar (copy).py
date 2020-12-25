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
