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