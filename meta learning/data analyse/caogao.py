import keras
from keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(360, 360, 3)))
model.add(layers.Conv2D(filters=32, kernel_size=11, strides=4, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu"))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(units=8, activation=None))
model.summary()
