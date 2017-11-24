#!/home/user0/anaconda3/bin/python
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import keras.backend as K
import numpy as np

'''
contractive autoencoder
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
encode = Dense(200, activation='relu')(input_img)
decode = Dense(784, activation='sigmoid')(encode)

autoencoder = Model(input=input_img, output=decode)

def contractive_loss(y_true, y_pred):
	# refer to keras/losses.py 
	cross_entropy = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)  # cross entropy
	mse = K.mean(K.square(y_pred-y_true), axis=-1)  # mse
	W = K.variable(value=autoencoder.get_layer('dense_1').get_weights()[0])
	W = K.transpose(W)
	h = autoencoder.get_layer('dense_1').output
	dh = h*(1-h)  # h=sigmoid(w*x+b) derivation=h(1-h)*W 
	contractive = 1e-4*K.sum(dh**2*K.sum(W**2, axis=1), axis=1)
	return cross_entropy + contractive

autoencoder.compile(optimizer='adam', loss=contractive_loss)
autoencoder.summary()

autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_test, x_test))


