#!/home/user0/anaconda3/bin/python
import keras
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

# train dataset and test dataset
x = np.arange(10000)/10000  # data normalization before training
np.random.shuffle(x)
y = 2*x + 3 + np.random.normal(0, 0.05, (10000))  # add noise
x_train, y_train = x[:7000], y[:7000]
x_test, y_test = x[7000:], y[7000:]

# model
a = Input(shape=(1,))
b = Dense(1)(a)
model = Model(input=a, output=b)
model.compile(optimizer='sgd', loss='mse')
model.summary()
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))

layer = model.get_layer('dense_1')
print('layer.get_weights: ', layer.get_weights())
print('layer.input: ', layer.input)
print('layper.output: ', layer.output)
print('layer.input_shape: ', layer.input_shape)
print('layer.output_shape: ', layer.output_shape)

