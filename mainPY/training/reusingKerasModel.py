import tensorflow as tf
from tensorflow import keras
import numpy as np

# ## Gradient Clipping
# All Keras optimizers accept `clipnorm` or `clipvalue` arguments:
optimizer = keras.optimizers.SGD(clipvalue=1.0)

optimizer = keras.optimizers.SGD(clipnorm=1.0)

# ## Reusing Pretrained Layers
# ### Reusing a Keras model
# Let's split the fashion MNIST training set in two:
# * `X_train_A`: all images of all items except for sandals and shirts (classes 5 and 6).
# * `X_train_B`: a much smaller training set of just the first 200 images of sandals or shirts.
# 
# The validation set and the test set are also split this way, but without restricting the number of images.
# 
# We will train a model on set A (classification task with 8 classes), and try to reuse it to tackle set B 
# (binary classification). We hope to transfer a little bit of knowledge from task A to task B, 
# since classes in set A (sneakers, ankle boots, coats, t-shirts, etc.) are somewhat similar to classes in set B 
# (sandals and shirts). However, since we are using `Dense` layers, only patterns that occur at the 
# same location can be reused (in contrast, convolutional layers will transfer much better, since learned 
# patterns can be detected anywhere on the image, as we will see in the CNN chapter).

def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

print(X_train_A.shape,X_train_B.shape,y_train_A[:30],y_train_B[:30] )

tf.random.set_seed(42)
np.random.seed(42)

def modelATraining():

    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_A.add(keras.layers.Dense(8, activation="softmax"))

    model_A.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])

    history = model_A.fit(X_train_A, y_train_A, epochs=20,
                        validation_data=(X_valid_A, y_valid_A))

    model_A.save("mainPY/training/model_A.h5")


def modelBTraining():

    model_B = keras.models.Sequential()
    model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_B.add(keras.layers.Dense(1, activation="sigmoid"))

    model_B.compile(loss="binary_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])

    history = model_B.fit(X_train_B, y_train_B, epochs=20,
                            validation_data=(X_valid_B, y_valid_B))

model_A = keras.models.load_model("mainPY/training/model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# model_A_clone = keras.models.clone_model(model_A)
# model_A_clone.set_weights(model_A.get_weights())


for layer in model_B_on_A.layers[:-1]:
        layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy",
                        optimizer=keras.optimizers.SGD(lr=1e-3),
                        metrics=["accuracy"])


history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                            validation_data=(X_valid_B, y_valid_B))


model_A = keras.models.load_model("mainPY/training/model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

model_B = keras.models.Sequential()
model_B_on_A.compile(loss="binary_crossentropy",
                        optimizer=keras.optimizers.SGD(lr=1e-3),
                        metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                            validation_data=(X_valid_B, y_valid_B))

# So, what's the final verdict?
model_B.evaluate(X_test_B, y_test_B)

model_B_on_A.evaluate(X_test_B, y_test_B)


def fasterOptimizers():

    # Fater Optimizers:
    # ## Momentum optimization
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # ## Nesterov Accelerated Gradient
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    # ## AdaGrad
    optimizer = keras.optimizers.Adagrad(lr=0.001)

    # ## RMSProp
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

    # ## Adam Optimization
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # ## Adamax Optimization
    optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)

    # ## Nadam Optimization
    optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)



# modelATraining()
# modelBTraining()