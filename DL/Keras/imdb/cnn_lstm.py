from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense


# embedding
max_features = 20000
maxlen = 200
embedding_dims = 128

# convolution
kernel_size = 5
filters = 64
pool_size = 4

# lstm
lstm_output_size = 70

# training
batch_size = 32
epochs = 5


# dataset
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('y_train type: ', type(y_train))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Conv1D(filters, kernel_size, padding='same', activation='relu', strides=1))
# comparing with cnn.py (only can using GlobalMaxPooling1D)
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))




