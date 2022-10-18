import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


#import pathlib
#dataset_url = "/Users/dominicduda/Downloads/Bike"
#data_dir = tf.keras.utils.get_file('bike_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)

pictures = '/Users/dominicduda/Downloads/Bike'

#Keras
batch_size = 32
img_height = 180
img_width = 180

image_datagen = ImageDataGenerator(rescale=1./255)

image_generator = image_datagen.flow_from_directory(
    pictures,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#image_count = len(list(image_generator('*/*.jpg')))
#print(image_count)

train_ds = tf.keras.utils.image_dataset_from_directory(
  pictures,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  pictures,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

model.save('/Users/dominicduda/Library/Mobile Documents/com~apple~CloudDocs/FH-Campus/AR_and_VR_Development/Development/halbesG/model')


