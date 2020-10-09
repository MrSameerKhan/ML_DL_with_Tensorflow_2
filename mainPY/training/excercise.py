import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt
K = keras.backend



# ## 8. Deep Learning on CIFAR10

# *Exercise: Build a DNN with 20 hidden layers of 100 neurons each 
# (that's too many, but it's the point of this exercise). 
# Use He initialization and the ELU activation function.*

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                    last_iterations=None, last_rate=None):
            self.iterations = iterations
            self.max_rate = max_rate
            self.start_rate = start_rate or max_rate / 10
            self.last_iterations = last_iterations or iterations // 10 + 1
            self.half_iteration = (iterations - self.last_iterations) // 2
            self.last_rate = last_rate or self.start_rate / 1000
            self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                    self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                    self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
        init_weights = model.get_weights()
        iterations = len(X) // batch_size * epochs
        factor = np.exp(np.log(max_rate / min_rate) / iterations)
        init_lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, min_rate)
        exp_lr = ExponentialLearningRate(factor)
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                            callbacks=[exp_lr])
        K.set_value(model.optimizer.lr, init_lr)
        model.set_weights(init_weights)
        return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                    activation="elu",
                                    kernel_initializer="he_normal"))

# ### b.
# *Exercise: Using Nadam optimization and early stopping, 
# train the network on the CIFAR10 dataset. You can load it with
#  `keras.datasets.cifar10.load_data()`.
#  The dataset is composed of 60,000 32 × 32–pixel color images 
# (50,000 for training, 10,000 for testing) with 10 classes, so you'll 
# need a softmax output layer with 10 neurons. Remember to search 
# for the right learning rate each time you change the model's architecture or hyperparameters.*

# Let's add the output layer to the model:


model.add(keras.layers.Dense(10, activation="softmax"))

# Let's use a Nadam optimizer with a learning rate of 5e-5. I tried 
# learning rates 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 and 1e-2, and I compared 
# their learning curves for 10 epochs each (using the TensorBoard callback, below). 
# The learning rates 3e-5 and 1e-4 were pretty good, so I tried 5e-5, which turned out to be slightly better.

optimizer = keras.optimizers.Nadam(lr=5e-5)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

# Let's load the CIFAR10 dataset. We also want to use early stopping, 
# so we need a validation set. Let's use the first 5,000 images of 
# the original training set as the validation set:

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]

# Now we can create the callbacks we need and train the model:

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/training/excercise_my_cifar10_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "mainPY/training/excercise_my_cifar10_logs", "run_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

# get_ipython().magic('tensorboard --logdir=./my_cifar10_logs --port=6006')

model.fit(X_train, y_train, epochs=1,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks)

model = keras.models.load_model("mainPY/training/excercise_my_cifar10_model.h5")
model.evaluate(X_valid, y_valid)

# The model with the lowest validation loss gets about 47% accuracy 
# on the validation set. It took 39 epochs to reach the lowest validation loss, 
# with roughly 10 seconds per epoch on my laptop (without a GPU).
#  Let's see if we can improve performance using Batch Normalization.

# ### c.
# *Exercise: Now try adding Batch Normalization and compare 
# the learning curves: Is it converging faster than before? 
# Does it produce a better model? How does it affect training speed?*

# The code below is very similar to the code above, with a few changes:
# 
# * I added a BN layer after every Dense layer (before the activation function), 
# except for the output layer. I also added a BN layer before the first hidden layer.
# * I changed the learning rate to 5e-4. I experimented with 
# 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3 and 3e-3, and 
# I chose the one with the best validation performance after 20 epochs.
# * I renamed the run directories to run_bn_* and the model file name to my_cifar10_bn_model.h5.


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
model.add(keras.layers.BatchNormalization())
for _ in range(20):
    model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("elu"))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(lr=5e-4)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/training/excercise_my_cifar10_bn_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "mainPY/training/excercise_my_cifar10_logs", "run_bn_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

model.fit(X_train, y_train, epochs=1,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks)

model = keras.models.load_model("mainPY/training/excercise_my_cifar10_bn_model.h5")
model.evaluate(X_valid, y_valid)

# * *Is the model converging faster than before?* Much faster! 
# The previous model took 39 epochs to reach the lowest validation loss,
#  while the new model with BN took 18 epochs. That's more than twice as
#  fast as the previous model. The BN layers stabilized training and 
# allowed us to use a much larger learning rate, so convergence was faster.
# * *Does BN produce a better model?* Yes! The final model is also 
# much better, with 55% accuracy instead of 47%. It's still not a
#  very good model, but at least it's much better than before 
# (a Convolutional Neural Network would do much better, but that's a different topic, see chapter 14).
# * *How does BN affect training speed?* Although the model 
# converged twice as fast, each epoch took about 16s instead of 10s, 
# because of the extra computations required by the BN layers. 
# So overall, although the number of epochs was reduced by 50%, 
# the training time (wall time) was shortened by 30%. Which is still pretty significant!

# ### d.
# *Exercise: Try replacing Batch Normalization with SELU, 
# and make the necessary adjustements to ensure the network 
# self-normalizes (i.e., standardize the input features, 
# use LeCun normal initialization, make sure the 
# DNN contains only a sequence of dense layers, etc.).*


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(lr=7e-4)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/training/excercise_my_cifar10_selu_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "mainPY/training/excercise_my_cifar10_logs", "run_selu_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

model.fit(X_train_scaled, y_train, epochs=1,
            validation_data=(X_valid_scaled, y_valid),
            callbacks=callbacks)

model = keras.models.load_model("mainPY/training/excercise_my_cifar10_selu_model.h5")
model.evaluate(X_valid_scaled, y_valid)

model = keras.models.load_model("mainPY/training/excercise_my_cifar10_selu_model.h5")
model.evaluate(X_valid_scaled, y_valid)

# We get 51.4% accuracy, which is better than the original model,
#  but not quite as good as the model using batch normalization. 
# Moreover, it took 13 epochs to reach the best model, 
# which is much faster than both the original model and the BN model, 
# plus each epoch took only 10 seconds, just like the original model. 
# So it's by far the fastest model to train (both in terms of epochs and wall time).
# ### e.
# *Exercise: Try regularizing the model with alpha dropout. 
# Then, without retraining your model, see if you can achieve better accuracy using MC Dropout.*

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))

model.add(keras.layers.AlphaDropout(rate=0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(lr=5e-4)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mainPY/training/excercise_my_cifar10_alpha_dropout_model.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "mainPY/training/excercise_my_cifar10_logs", "run_alpha_dropout_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

model.fit(X_train_scaled, y_train, epochs=1,
            validation_data=(X_valid_scaled, y_valid),
            callbacks=callbacks)

model = keras.models.load_model("mainPY/training/excercise_my_cifar10_alpha_dropout_model.h5")
model.evaluate(X_valid_scaled, y_valid)

# The model reaches 50.8% accuracy on the validation set.
#  That's very slightly worse than without dropout (51.4%).
#  With an extensive hyperparameter search, it might be possible
#  to do better (I tried dropout rates of 5%, 10%, 20% and 40%, 
# and learning rates 1e-4, 3e-4, 5e-4, and 1e-3), but probably not much better in this case.

# Let's use MC Dropout now. We will need the `MCAlphaDropout`
#  class we used earlier, so let's just copy it here for convenience:

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# Now let's create a new model, identical to the one we just
#  trained (with the same weights), but with `MCAlphaDropout` dropout layers instead of `AlphaDropout` layers:

mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])

# Then let's add a couple utility functions. The first will run the model
#  many times (10 by default) and it will return the mean predicted class 
# probabilities. The second will use these mean probabilities to predict the most likely class for each instance:

def mc_dropout_predict_probas(mc_model, X, n_samples=10):
    Y_probas = [mc_model.predict(X) for sample in range(n_samples)]
    return np.mean(Y_probas, axis=0)

def mc_dropout_predict_classes(mc_model, X, n_samples=10):
    Y_probas = mc_dropout_predict_probas(mc_model, X, n_samples)
    return np.argmax(Y_probas, axis=1)


# Now let's make predictions for all the instances in the validation set,
#  and compute the accuracy:


y_pred = mc_dropout_predict_classes(mc_model, X_valid_scaled)
accuracy = np.mean(y_pred == y_valid[:, 0])
accuracy


# We only get virtually no accuracy improvement in this case (from 50.8% to 50.9%).
# 
# So the best model we got in this exercise is the Batch Normalization model.

# ### f.
# *Exercise: Retrain your model using 1cycle scheduling and 
# see if it improves training speed and model accuracy.*


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))

model.add(keras.layers.AlphaDropout(rate=0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(lr=1e-3)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

batch_size = 128
rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=batch_size)
plot_lr_vs_loss(rates, losses)
plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 1.4])

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                    kernel_initializer="lecun_normal",
                                    activation="selu"))

model.add(keras.layers.AlphaDropout(rate=0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(lr=1e-2)
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

n_epochs = 1
onecycle = OneCycleScheduler(len(X_train_scaled) // batch_size * n_epochs, max_rate=0.05)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[onecycle])

# One cycle allowed us to train the model in just 15 epochs, each taking only 3 seconds (thanks to the larger batch size). This is over 3 times faster than the fastest model we trained so far. Moreover, we improved the model's performance (from 50.8% to 52.8%). The batch normalized model reaches a slightly better performance, but it's much slower to train.

