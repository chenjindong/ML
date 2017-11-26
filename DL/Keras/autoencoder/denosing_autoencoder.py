#!/home/user0/anaconda3/bin/python
import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

'''
denoising autoencoder
obvious results are shown by matplotlib
'''
hidden_dim = 64
batch_size = 128
epochs = 5

# dataset
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)  # (60000, 28*28) (10000, 28*28)

x_train_noisy = x_train + np.random.normal(0, 0.5, x_train.shape)
x_test_noisy = x_test + np.random.normal(0, 0.5, x_test.shape)  # normal distribution
x_train_noisy = np.clip(x_train_noisy, 0, 1)  # input belong to [0, 1]
x_test_noisy = np.clip(x_test_noisy, 0, 1)

# model
input_img = Input(shape=(28*28,))
encode = Dense(hidden_dim, activation='relu')(input_img)
decode = Dense(784, activation='sigmoid')(encode)

autoencoder = Model(input=input_img, output=decode)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test_noisy, x_test))

# predict and result
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()