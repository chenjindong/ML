from keras.models import Sequential
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, Dense, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D

max_features = 20000
maxlen = 400
embedding_dims = 128
filters = 250
kernel_size = 3
hidden_dims = 250
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

# model
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))
# padding ='same' => the output has the same length as the original input; padding='valid' => no padding
model.add(Conv1D(filters, kernel_size, padding='same', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
# we can also use next two lines instead of GlobalMaxPooling1D
# model.add(MaxPooling1D(pool_size=4))
# model.add(Flatten())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))



