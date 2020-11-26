# %%
# Functional API

# Week 1 3 Coding Tutorials


from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model

inputs = Input(shape=(32,1))
h = Conv1D(16,5, activation="relu")(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()
outputs = Dense(20, activation="sigmoid")(h)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss="binar_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=20)

test_loss, test_acc = model.evaluate[X_test, y_test]

preds = model.predict(X_sample)



# %%

# Multiple Inputs and Outputs

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate
from tensorflow.keras.models import Model

inputs = Input(shape=(32,1), name="inputs")
h = Conv1D(16,5, activation="relu")(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
aux_inputs = Input(shape=(12,), name="aux_inputs")
h = Concatenate()([h, aux_inputs])
outputs = Dense(20, activation="sigmoid", name="outputs")(h)
aux_outputs = Dense(1, activation="linear", name ="aux_outputs")(h)

model = Model(inputs=[inputs, aux_inputs], outputs=[outputs, aux_outputs])

model.compile(loss={"outputs":"binary_crossentropy", "aux_outputs":"mse"},
                loss_weighs={"outputs":1 , "aux_outputs": 0.4}, metrics = ["accuracy"])


history = model.fit({"inputs":x_train, "aux_inputs": x_aux},
                    {"outputs":y_train, "aux_outputs": y_aux},
                    validation_split=0.2, epochs=20)

# %%

# Tutorial Multiple Inputs and Outputs


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

pd_dat = pd.read_csv("data/diagnosis.csv")
dataset = pd_dat.values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,:6], dataset[:,6:], test_size=0.33)

temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train = np.transpose(X_train)
temp_test, nocc_test, lumb_test, up_test, mict_test, bis_test = np.transpose(X_test)

inflam_train, nephr_train = Y_train[:,0], Y_train[:,1]
inflam_test, nephr_test = Y_test[:,0], Y_test[:,1]


from tensorflow.keras import Input, layers

shape_inputs = (1,)
temperature = Input(shape=shape_inputs, name="temp")
nausea_occurence = Input(shape=shape_inputs, name="nocc")
lumbar_pain = Input(shape=shape_inputs, name="lumbp")
urine_pushing = Input(shape=shape_inputs, name="up")
micturition_pains = Input(shape=shape_inputs, name="mict")
bis = Input(shape=shape_inputs, name="bis")

list_inputs = [temperature, nausea_occurence, lumbar_pain, urine_pushing, micturition_pains, bis]

x = layers.concatenate(list_inputs)

inflammation_pred = layers.Dense(1, activation="sigmoid", name="inflam")(x)
nephritis_pred = layers.Dense(1, activation="sigmoid", name="nephr")(x)

list_outputs = [inflammation_pred, nephritis_pred]

model = tf.keras.Model(inputs=list_inputs, outputs=list_outputs)

tf.keras.utils.plot_model(model, "multi_input_output_model.png", show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
            loss=["binary_crossentropy", "binary_crossentropy"],
                metrics={"inflam" : ["acc"],
                    "nephr":["acc"]},
                    loss_weights= [1., 0.2])


inputs_train = {"temp": temp_train, "nocc": nocc_train, "lumbp": lumbp_train, # Recommended dict if more than one input
                "up": up_train, "mict" : mict_train, "bis" : bis_train}
outputs_train = {"inflam":inflam_train, "nephr": nephr_train}

# inputs_train = [temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train]
# outputs_train = [inflam_train, nephr_train]


history = model.fit(inputs_train, outputs_train, epochs=1000, batch_size=128, verbose=False)


acc_keys = [k for k in history.history.keys() if k in ("inflam_acc", "nephr_acc")]
loss_keys = [k for k in history.history.keys() if not k in acc_keys]

for k , v in history.history.items():
    if k in acc_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.figure(1)
plt.title("Accuracy vs epochs")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(acc_keys, loc="upper right")

plt.figure(2)
plt.title("Loss vs epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loss_keys, loc="upper right")

plt.show()


model.evaluate([temp_test, nocc_test, lumb_test, up_test, mict_test, bis_test],
                [inflam_test, nephr_test], verbose=2)



# %%

# Variables

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential([

        Dense(1, input_shape=(4,))
])

print(model.weights)


import tensorflow as tf

my_var = tf.Variable([-1,2], dtype=tf.float32, name="my_var")
my_var.assign([3.5, -1.1])

x = my_var.numpy()



# %%

# Tensors

import tensorflow as tf

my_var = tf.Variable([-1,2], dtype=tf.float32, name="my_var")
h = my_var + [5, 4]

print(h)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

inputs = Input(shape=(5,))
h = Dense(16, activation="sigmoid")(inputs)
outputs = Dense(10, activation="softmax")(h)

model = Model(inputs= inputs, outputs=outputs)

print(model.input)
print(model.output)



x = tf.constant([
        [5,2],
        [1,3]
])

x_arr = x.numpy()


x = tf.ones(shape=(2,1))

y = tf.zeros(shape=(2,1))

# %%

# Tutorial Variables and Tensors





# %%

# Accessing layer Variables


from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model

inputs = Input(shape=(32,1))
h = Conv1D(3,5, activation="relu")(inputs)
h = AveragePooling1D(3)(h)
h = Flatten()(h)
outputs = Dense(20, activation="sigmoid")(h)

model = Model(inputs=inputs, outputs=outputs)

# print(model.layers)
# print(model.layers[1])
# print(model.layers[1].weights)
print(model.layers[1].get_weights())
print(model.layers[1].kernel)
print(model.layers[1].bias)




inputs = Input(shape=(32,1), name="input_layer")
h = Conv1D(3,5, activation="relu", name="conv1d_layer")(inputs)
h = AveragePooling1D(3, name="abg_pool1d_layer")(h)
h = Flatten(name="flatten_layer")(h)
outputs = Dense(20, activation="sigmoid", name="dense_layer")(h)

model = Model(inputs=inputs, outputs=outputs)

print(model.get_layer("conv1d_layer").bias)




# %%

# Accessing layer Tensors

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model

inputs = Input(shape=(32,1), name="input_layer")
h = Conv1D(3,5, activation="relu", name="conv1d_layer")(inputs)
h = AveragePooling1D(3, name="abg_pool1d_layer")(h)
h = Flatten(name="flatten_layer")(h)
outputs = Dense(20, activation="sigmoid", name="dense_layer")(h)

model = Model(inputs=inputs, outputs=outputs)

print(model.get_layer("conv1d_layer").input)
print(model.get_layer("conv1d_layer").output)


flatten_output = model.get_layer("flatten_layer".output)

model2 = Model(inputs=inputs, outputs=flatten_output)

new_outputs = Dense(10, activation="softmax")(model2.output)
model3 = Model(inputs=model2.input, outputs= new_outputs)

# %%

# Tutorial Accessing model layers



# %%

# Freezing layers

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model, load_model

inputs = Input(shape=(8,81), name="input_layer")
h = Conv2D(16,3, activation="relu", name="conv2d_layer")(inputs)
h = MaxPooling2D(3, name="max_pool2d_layer")(h)
h = Flatten(name="flatten_layer")(h)
outputs = Dense(10, activation="softmax", name="softmax_layer")(h)

model = Model(inputs=inputs, outputs=outputs)

model.get_layer("conv2d_layer").trainable= False

model.compile(loss="cross_categorical_crossentropy")

# Load
model = load_model("my_pretrained_model")
model.trainable = False

flatten_output = model.get_layer("flatten_layer").output
new_outputs = Dense(5, activation="softmax", name="new_softmax_layer")(flatten_output)

new_model = Model(inputs=model.input, outputs=new_outputs)
new_model.fit(X_train, y_train, epochs=10)



# %%

# Tutorial Freezing layers



# %%

# Programming assignment: Transfer learning




