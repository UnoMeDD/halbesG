#main.py>

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from main import class_names

model = keras.models.load_model('/Users/dominicduda/Library/Mobile Documents/com~apple~CloudDocs/FH-Campus/AR_and_VR_Development/Development/halbesG/')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

img_height = 180
img_width = 180

#class_names = model

bike_url = "https://powerbronze.de/media/image/6b/7b/22/410-y113-bike-1AdpodXk4hSUOu.jpg"
bike_path = tf.keras.utils.get_file('FZS', origin=bike_url)

img = tf.keras.utils.load_img(
    bike_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
 "This image most likely belongs to {} with a {:.2f} percent confidence."
   .format(class_names[np.argmax(score)], 100 * np.max(score))
)
