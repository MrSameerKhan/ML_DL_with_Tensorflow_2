import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
import tensorflow as tf


vgg_model = VGG19()

# from tensorflow.keras.models import load_model
# vgg_model = load_model(models/Vgg19.h5)

vgg_input = vgg_model.input
vgg_layers = vgg_model.layers
print(vgg_model.summary())

from tensorflow.keras.models import Model

layer_outputs = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input, outputs=layer_outputs)

tf.keras.utils.plot_model(features, "vgg19_model.png", show_shapes=True)

img = np.random.random((1,224,224,3)).astype("float32")
extracted_features = features(img)

import IPython.display as display
from PIL import Image

display.display(Image.open("data/cool_cat.jpg"))

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

img_path = "data/cool_cat.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)