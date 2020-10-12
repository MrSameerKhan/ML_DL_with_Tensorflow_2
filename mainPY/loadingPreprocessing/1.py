
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from person_pb2 import Person
from sklearn.datasets import load_sample_images
import numpy as np
import os
import matplotlib.pyplot as plt

# ## Split the California dataset to multiple CSV files
# Let's start by loading and preparing the California housing dataset.
#  We first load it, then split it into a training set, 
# a validation set and a test set, and finally we scale it:

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_


# For a very large dataset that does not fit in memory, 
# you will typically want to split it into many files first, 
# then have TensorFlow read these files in parallel. 
# To demonstrate this, let's start by splitting the housing 
# dataset and save it to 20 CSV files:

def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)


# Okay, now let's take a peek at the first few lines of one of 
# these CSV files:

pd.read_csv(train_filepaths[0]).head()

with open(train_filepaths[0]) as f:
    for i in range(5):
        print(f.readline(), end="")

print(train_filepaths)

# ## Building an Input Pipeline
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

for filepath in filepath_dataset:
    print(filepath)

n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)

for line in dataset.take(5):
    print(line.numpy())

# Notice that field 4 is interpreted as a string.
record_defaults=[0, np.nan, tf.constant(np.nan, dtype=tf.float64), "Hello", tf.constant([])]
parsed_fields = tf.io.decode_csv('1,2,3,4,5', record_defaults)
print(parsed_fields)

# Notice that all missing fields are replaced with their
#  default value, when provided:
parsed_fields = tf.io.decode_csv(',,,,5', record_defaults)
print(parsed_fields)

# The 5th field is compulsory (since we provided `tf.constant([])` 
# as the "default value"), so we get an exception if 
# we do not provide it:

try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

# The number of fields should match exactly 
# the number of fields in the `record_defaults`:

try:
    parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

n_inputs = 8 # X_train.shape[-1]

@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y



    preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')



def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

tf.random.set_seed(42)

train_set = csv_reader_dataset(train_filepaths, batch_size=3)
for X_batch, y_batch in train_set.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
            validation_data=valid_set)



model.evaluate(test_set, steps=len(X_test) // batch_size)



new_set = test_set.map(lambda X, y: X) # we could instead just pass test_set, Keras would ignore the labels
X_new = X_test
model.predict(new_set, steps=len(X_new) // batch_size)


optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error

n_epochs = 5
batch_size = 32
n_steps_per_epoch = len(X_train) // batch_size
total_steps = n_epochs * n_steps_per_epoch
global_step = 0
for X_batch, y_batch in train_set.take(total_steps):
    global_step += 1
    print("\rGlobal step {}/{}".format(global_step, total_steps), end="")
    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
        loss = tf.add_n([main_loss] + model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error

@tf.function
def train(model, n_epochs, batch_size=32,
            n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
        train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                        n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                        n_parse_threads=n_parse_threads, batch_size=batch_size)
        for X_batch, y_batch in train_set:
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

train(model, 5)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error

# @tf.function
# def train(model, n_epochs, batch_size=32,
#             n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
#         train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
#                         n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
#                         n_parse_threads=n_parse_threads, batch_size=batch_size)
#     n_steps_per_epoch = len(X_train) // batch_size
#     total_steps = n_epochs * n_steps_per_epoch
#     global_step = 0
#     for X_batch, y_batch in train_set.take(total_steps):
#         global_step += 1
#         if tf.equal(global_step % 100, 0):
#             tf.print("\rGlobal step", global_step, "/", total_steps)
#         with tf.GradientTape() as tape:
#             y_pred = model(X_batch)
#             main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
#             loss = tf.add_n([main_loss] + model.losses)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# train(model, 5)

# Here is a short description of each method in the `Dataset` class:

for m in dir(tf.data.Dataset):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(tf.data.Dataset, m)
        if hasattr(func, "__doc__"):
            print("● {:21s}{}".format(m + "()", func.__doc__.split("\n")[0]))


# ## The `TFRecord` binary format
# A TFRecord file is just a list of binary records. 
# You can create one using a `tf.io.TFRecordWriter`:

with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

# And you can read it using a `tf.data.TFRecordDataset`:
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)


# You can read multiple TFRecord files with just one 
# `TFRecordDataset`. By default it will read them one at a time,
#  but if you set `num_parallel_reads=3`, it will read 3 at a 
# time in parallel and interleave their records:

filepaths = ["my_test_{}.tfrecord".format(i) for i in range(5)]
for i, filepath in enumerate(filepaths):
    with tf.io.TFRecordWriter(filepath) as f:
        for j in range(3):
            f.write("File {} record {}".format(i, j).encode("utf-8"))

dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=3)
for item in dataset:
    print(item)


options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                    compression_type="GZIP")
for item in dataset:
    print(item)

# ### A Brief Intro to Protocol Buffers
# For this section you need to [install protobuf]
# (https://developers.google.com/protocol-buffers/docs/downloads).
#  In general you will not have to do so when using TensorFlow, 
# as it comes with functions to create and parse protocol buffers 
# of type `tf.train.Example`, which are generally sufficient. 
# However, in this section we will learn about protocol buffers 
# by creating our own simple protobuf definition, 
# so we need the protobuf compiler (`protoc`): 
# we will use it to compile the protobuf definition to a 
# Python module that we can then use in our code.

# First let's write a simple protobuf definition:

# And let's compile it (the `--descriptor_set_out` and
#  `--include_imports` options are only required for the
#  `tf.io.decode_proto()` example below):

person = Person(name="Al", id=123, email=["a@b.com"])  # create a Person
print(person)  # display the Person

person.name  # read a field

person.name = "Alice"  # modify a field

person.email[0]  # repeated fields can be accessed like arrays

person.email.append("c@d.com")  # add an email address

s = person.SerializeToString()  # serialize to a byte string
s

person2 = Person()  # create a new Person
person2.ParseFromString(s)  # parse the byte string (27 bytes)

person == person2  # now they are equal

# In rare cases, you may want to parse a custom protobuf 
# (like the one we just created) in TensorFlow. 
# For this you can use the `tf.io.decode_proto()` function:

person_tf = tf.io.decode_proto(
    bytes=s,
    message_type="Person",
    field_names=["name", "id", "email"],
    output_types=[tf.string, tf.int32, tf.string],
    descriptor_source="person.desc")

person_tf.values

# ### TensorFlow Protobufs
# Here is the definition of the tf.train.Example protobuf:

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())

feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)


parsed_example

parsed_example

parsed_example["emails"].values[0]

tf.sparse.to_dense(parsed_example["emails"], default_value=b"")

parsed_example["emails"].values

# ### Putting Images in TFRecords



img = load_sample_images()["images"][0]
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()

data = tf.io.encode_jpeg(img)
example_with_image = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))
serialized_example = example_with_image.SerializeToString()
# then save to TFRecord

feature_description = { "image": tf.io.VarLenFeature(tf.string) }
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_jpeg(example_with_image["image"].values[0])


# Or use `decode_image()` which supports BMP, GIF, JPEG and 
# PNG formats:

decoded_img = tf.io.decode_image(example_with_image["image"].values[0])

plt.imshow(decoded_img)
plt.title("Decoded Image")
plt.axis("off")
plt.show()

# ### Putting Tensors and Sparse Tensors in TFRecords
# Tensors can be serialized and parsed easily using `tf.io.serialize_tensor()` and `tf.io.parse_tensor()`:

t = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
s = tf.io.serialize_tensor(t)
s

tf.io.parse_tensor(s, out_type=tf.float32)

serialized_sparse = tf.io.serialize_sparse(parsed_example["emails"])
serialized_sparse

BytesList(value=serialized_sparse.numpy())

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(10)
for serialized_examples in dataset:
    parsed_examples = tf.io.parse_example(serialized_examples,
                                            feature_description)

parsed_examples
