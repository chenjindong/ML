#!/home/user0/anaconda3/bin/python
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt

'''
deep autoencoder
'''

hidden1_dim = 128
hidden2_dim = 64
hidden3_dim = 32
batch_size = 128
epochs = 5

# dataset
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
x_train = x_train/255.0
x_test = x_test/255.0
#print(x_train.dtype)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape, x_test.shape)  # (60000, 28*28) (10000, 28*28)

# model
input_img = Input(shape=(28*28,))
encode = Dense(hidden1_dim, activation='relu')(input_img)
encode = Dense(hidden2_dim, activation='relu')(encode)
encode = Dense(hidden3_dim, activation='relu')(encode)
decode = Dense(hidden2_dim, activation='relu')(encode)
decode = Dense(hidden1_dim, activation='relu')(decode)
decode = Dense(784, activation='sigmoid')(decode)

autoencoder = Model(input=input_img, output=decode)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test, x_test))

# predict and result
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
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