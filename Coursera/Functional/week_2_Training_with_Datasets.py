# %%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import tensorflow as tf
print(tf.__version__)


# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# %% [markdown]
# #### Load the UCI Bank Marketing Dataset
# %% [markdown]
# #### Import the data
# 
# The dataset required for this tutorial can be downloaded from the following link:
# 
# https://drive.google.com/open?id=1cNtP4iDyGhF620ZbmJdmJWYQrRgJTCum
# 
# You should store these files in Drive for use in this Colab notebook.

# %%
# Run this cell to connect to your Drive folder

# from google.colab import drive
# drive.mount('/content/gdrive')


# %%
# Load the CSV file into a pandas DataFrame

bank_dataframe = pd.read_csv('data/bank/bank-full.csv', delimiter=';')


# %%
# Show the head of the DataFrame

bank_dataframe.head()


# %%
# Print the shape of the DataFrame

print(bank_dataframe.shape)


# %%
# Select features from the DataFrame

features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
            'loan', 'contact', 'campaign', 'pdays', 'poutcome']
labels = ['y']

bank_dataframe = bank_dataframe.filter(features + labels)


# %%
# Show the head of the DataFrame

bank_dataframe.head()

# %% [markdown]
# #### Preprocess the data

# %%
# Convert the categorical features in the DataFrame to one-hot encodings

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
categorical_features = ['default', 'housing', 'job', 'loan', 'education', 'contact', 'poutcome']

for feature in categorical_features:
    bank_dataframe[feature] = tuple(encoder.fit_transform(bank_dataframe[feature]))


# %%
# Show the head of the DataFrame

bank_dataframe.head()


# %%
# Shuffle the DataFrame

bank_dataframe = bank_dataframe.sample(frac=1).reset_index(drop=True)

# %% [markdown]
# #### Create the Dataset object

# %%
# Convert the DataFrame to a Dataset

bank_dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))


# %%
# Inspect the Dataset object

bank_dataset.element_spec

# %% [markdown]
# #### Filter the Dataset

# %%
# First check that there are records in the dataset for non-married individuals

def check_divorced():
    bank_dataset_iterable = iter(bank_dataset)
    for x in bank_dataset_iterable:
        if x['marital'] != 'divorced':
            print('Found a person with marital status: {}'.format(x['marital']))
            return
    print('No non-divorced people were found!')

check_divorced()


# %%
# Filter the Dataset to retain only entries with a 'divorced' marital status

bank_dataset = bank_dataset.filter(lambda x : tf.equal(x['marital'], tf.constant([b'divorced']))[0] )


# %%
# Check the records in the dataset again

check_divorced()

# %% [markdown]
# #### Map a function over the dataset

# %%
# Convert the label ('y') to an integer instead of 'yes' or 'no'

def map_label(x):
    x['y'] = 0 if (x["y"] == tf.constant([b"no"], dtype = tf.string)) else 1
    return x

bank_dataset = bank_dataset.map(map_label)


# %%
# Inspect the Dataset object

bank_dataset.element_spec


# %%
# Remove the 'marital' column

bank_dataset = bank_dataset.map(lambda x : {key:val for key , val in x.items() if  key != "martial"})


# %%
# Inspect the Dataset object

bank_dataset.element_spec

# %% [markdown]
# #### Create input and output data tuples

# %%
# Create an input and output tuple for the dataset

def map_feature_label(x):
    features = [[x['age']], [x['balance']], [x['campaign']], x['contact'], x['default'],
                x['education'], x['housing'], x['job'], x['loan'], [x['pdays']], x['poutcome']]
    return (tf.concat(features, axis=0), x['y'])


# %%
# Map this function over the dataset

bank_dataset = bank_dataset.Map(map_feature_label)


# %%
# Inspect the Dataset object

bank_dataset.element_spec

# %% [markdown]
# #### Split into a training and a validation set

# %%
# Determine the length of the Dataset

dataset_length = 0
for _ in bank_dataset:
    dataset_length += 1
print(dataset_length)


# %%
# Make training and validation sets from the dataset

training_elements = int( dataset_length = 0.7)
train_dataset = bank_dataset.take(training_elements)
validation_dataset = bank_dataset.skip(training_elements)

# %% [markdown]
# #### Build a classification model
# 
# Now let's build a model to classify the features.

# %%
# Build a classifier model

from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras import Sequential

model = Sequential()
model.add(Input(shape=(30,)))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization(momentum=0.8))
model.add(Dense(1, activation='sigmoid'))


# %%
# Compile the model

optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# %%
# Show the model summary

model.summary()

# %% [markdown]
# #### Train the model

# %%
# Create batched training and validation datasets

train_dataset = train_dataset.batch(20, drop_remainder = True)
validation_dataset = validation_dataset.batch(100)


# %%
# Shuffle the training data

train_dataset = train_dataset.shuffle(1000)


# %%
# Fit the model

history = model.fit(train_dataset,validation_dataset, epochs =5)


# %%
# Plot the training and validation accuracy

plt.Plot(history.epoch, history.history["accuracy"], label="training")
plt.plot(history.epoch, history.history["val_accuracy"], label= "validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

