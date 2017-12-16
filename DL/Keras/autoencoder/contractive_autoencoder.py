#!/home/user0/anaconda3/bin/python
from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy, mean_squared_error
import keras.backend as K
import matplotlib.pyplot as plt

'''
contractive autoencoder
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

# model
input_img = Input(shape=(28*28,))
encode = Dense(hidden_dim, activation='relu', name='hidden_layer')(input_img)
decode = Dense(784, activation='sigmoid')(encode)

autoencoder = Model(input=input_img, output=decode)

# 自定义loss function
def contractive_loss(y_true, y_pred):
    cross_entropy = binary_crossentropy(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    W = K.variable(value=autoencoder.get_layer('hidden_layer').get_weights()[0])  # pay attention to
    W = K.transpose(W)
    h = autoencoder.get_layer('hidden_layer').output
    dh = h*(1-h)  # h=sigmoid(w*x+b) derivation=h(1-h)*W
    contractive = 1e-4*K.sum(dh**2*K.sum(W**2, axis=1), axis=1)
    return cross_entropy + contractive

autoencoder.compile(optimizer='adam', loss=contractive_loss)
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
