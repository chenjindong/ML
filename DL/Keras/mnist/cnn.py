#!/home/user0/anaconda3/bin/python
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard

# Finally, we'll import some utilities. This will help us transform our data later:
from keras.utils import np_utils

from keras.datasets import mnist
import time

start = time.time()
batch_size = 32
epochs = 10

# dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# model
model = Sequential()

# L1, the input layer:
model.add(BatchNormalization(input_shape=(28, 28, 1)))
model.add(Convolution2D(32, (6, 6),  padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



# L2, the first hidden layer:
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='/tmp/keras_log')

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[tensorboard])
end = time.time()
print('training time (min):', (end - start)/60.)

print('done!')