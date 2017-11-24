#!/home/user0/anaconda3/bin/python
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
import numpy as np

'''
sparse autoencoder
'''

(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
x_train = x_train/255.0
x_test = x_test/255.0
#print(x_train.dtype)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)  # (60000, 28*28) (10000, 28*28)

input_img = Input(shape=(28*28,))
encode = Dense(200, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
decode = Dense(784, activation='sigmoid')(encode)

autoencoder = Model(input=input_img, output=decode)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_test, x_test))


