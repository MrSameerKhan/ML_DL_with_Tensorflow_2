

# %%

# Keras datasets

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# ~/.keras/datasets/mnist.npz

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# ~/.keras/datasets/cifar-10-batches-py
# ~/.keras/datasets/cifar-10-batches-py.tar.gz

from tensorflow.keras.datasets import imdb

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=1000, max_len=100)


# %%

# Tutorial Keras Datasets



# %%

# Dataset generators


def text_file_reader(filepath):
    with open(filepath, "r") as f:
        for row in f:
            yield row

text_datagen = text_file_reader("data_file.txt")

next(text_datagen) # "A line of text\n"
next(text_datagen) # "Another line of text\n"


import numpy as np

def get_data(batch_size):
    while True:
        y_train = np.random.choice([0,1], (batch_size, 1))
        x_train = np.random.choice(batch_size, 1) + (2 * y_train -1)
        yield x_train, y_train

datagen = get_data(32)
x, y = next(datagen;)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential([ Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer="sgd")

model.fit_generator(datagen, steps_per_epoch=1000, epochs=10)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential([ Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer="sgd")

 for _ in range(1000):
     x_train, y_train = next(datagen)
     model.train_on_batch(x_train, y_train)


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential([ Dense(1, activation="sigmoid")])
model.compile(loss="binary_crossentropy", optimizer="sgd")

model.fit_generator(datagen, steps_per_epoch=1000, epochs=10)

model.evaluate_generator(datagen_eval, steps=100)


# %%

# Keras Image Data Augmentation


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train),(x_test, y_test) = cifar10.load_data()

image_data_gen = ImageDataGenerator(rescale=None, horizontal_flip=True, 
                height_shift_range=0.2, fill_mode="nearest",
                    featurewise_center=True)

image_data_gen.fit(x_train)

train_datagen = image_data.gen.flow(x_train, y_train, batch_size=16)

model.fit_generator(train_datagen, epochs=20)


# %%

# Dataset Class

import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices([1,2 , 3, 4, 5, 6])

print(dataset)

for elem in dataset:
    print(elem)

for elem in dataset:
    print(elem.numpy())

dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([128,5]))

print(dataset.element_spec)


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([256,4]), minval=1,maxval=10, dtype=tf.int32), 
    tf.random.normal([256]))

print(dataset.element_spec)


for elem in dataset.take(2):
    print(elem)
    

from tensorflow.keras.datasets import cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

print(dataset.element_spec)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_datagen = ImageDataGenerator(width_shift_range=0.2, horizontal_flip=True)

dataset = tf.data.Dataset.from_generator(image_datagen.flow, args=[x_train, y_train],
    output_shapes=(tf.float32, tf.int32),
    output_shapes=([32, 32, 32, 3], [32, 1]))


# %%

# Training with Datasets


import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

print(data.element_spec)


(x_train, y_train),(x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

def rescale(image, label):
    return image/255, label

def label_filter(image, label):
    return tf.squeeze(label) != 9

dataset = dataset.map(rescale)
dataset = dataset.filter(label_filter)

dataset = dataset.shuffle(100)
dataset = dataset.batch(16, drop_remainder=True)
dataset = dataset.repeat()

history = model.fit(dataset, steps_per_epoch=x_train.shape[0]//16, epochs=10)





