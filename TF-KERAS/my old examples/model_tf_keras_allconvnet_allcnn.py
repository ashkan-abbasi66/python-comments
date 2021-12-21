"""

ALL-CNN-C (see table 2 in the following table)

Springenberg, Jost Tobias, Alexey Dosovitskiy, Thomas Brox,
and Martin Riedmiller. "Striving for simplicity:
The all convolutional net." arXiv preprint arXiv:1412.6806 (2014).


Found at
https://github.com/MateLabs/All-Conv-Keras/blob/master/allconv.py

"""

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(layers.Convolution2D(96, 3, padding = 'same', input_shape=(3, 32, 32)))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(96, 3,padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(96, 3, padding='same', strides = 2))
# model.add(layers.Dropout(0.5))

model.add(layers.Convolution2D(192, 3, padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(192, 3, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(192, 3,padding='same', strides = 2))
# model.add(layers.Dropout(0.5))

model.add(layers.Convolution2D(192, 3, padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(192, 1, padding='valid'))
model.add(layers.Activation('relu'))
model.add(layers.Convolution2D(10, 1, padding='valid'))


model.add(layers.GlobalAveragePooling2D())
model.add(layers.Activation('softmax'))

model.summary()