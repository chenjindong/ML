from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

'''
https://keras.io/getting-started/functional-api-guide/
Multi-input and multi-output models
'''
vocab_size = 10000
embedding_dims = 512

main_input = Input(shape=(100,), name='main_input')
x = Embedding(input_dim=vocab_size, output_dim=embedding_dims)(main_input)
lstm_out = LSTM(units=32)(x)
auxiliary_input = Input(shape=(5,), name='auxiliary_input')
x = concatenate([lstm_out, auxiliary_input])
x = Dense(units=64, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
x = Dense(units=64, activation='relu')(x)
main_output = Dense(units=1, activation='sigmoid', name='main_output')(x)
auxiliary_output = Dense(units=1, activation='sigmoid', name='auxiliary_output')(lstm_out)
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
