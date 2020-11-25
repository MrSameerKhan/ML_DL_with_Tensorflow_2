# %%

# The Functional API

from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, Concatenate
from tensorflow.keras.layers import Model

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

# Multiple Inputs and Outputs



