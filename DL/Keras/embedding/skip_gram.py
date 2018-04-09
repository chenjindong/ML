import cjdpy
import time
from keras.models import Model
from keras.layers import *
from keras.preprocessing.sequence import skipgrams
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


'''
skip-gram algorithm using negative sampling
'''

start = time.clock()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

# parameters
embedding_dims = 300
batch_size = 32
epochs = 20
path_windows = r'\\10.141.208.22\data\Chinese_isA\corpus\wikicorpus_seg.txt'
path_linux = 'wikicorpus_seg.txt'

# dataset
texts, vocabulary = [], []

data = cjdpy.load_csv(path_windows)
# data = cjdpy.load_csv(path_linux)

for i, item in enumerate(data):
    if i > 10000: break
    vocabulary += list(item)
    texts.append(item)

vocabulary = set(vocabulary)
vocab_size = len(vocabulary)
word2id, id2word = {}, {}
for idx, word in enumerate(vocabulary):
    word2id[word] = idx
    id2word[idx] = word

text_seq = [word2id[word] for text in texts for word in text]

X_train, y_train = skipgrams(text_seq, len(word2id), window_size=4)
X_train, y_train = np.array(X_train), np.array(y_train)

print('show 5 training examples...')
for i in range(5):
    print(X_train[i], y_train[i])


def build_skipgram_model():
    input1 = Input(shape=(1,))
    embeddings1 = Embedding(vocab_size, embedding_dims)(input1)
    output1 = Flatten()(embeddings1)

    input2 = Input(shape=(1,))
    embeddings2 = Embedding(vocab_size, embedding_dims)(input2)
    output2 = Flatten()(embeddings2)

    dot = Dot(axes=-1)([output1, output2])
    output = Dense(units=1, activation='sigmoid')(dot)
    model = Model(inputs=(input1, input2), outputs=output)
    return model


m = build_skipgram_model()
m.compile(optimizer='adam', loss='binary_crossentropy')
m.summary()

modelCheckpoint = ModelCheckpoint('skip_gram_model.h5', save_best_only=True, save_weights_only=True)

m.fit(x=[X_train[:, 0], X_train[:, 1]], y=y_train, batch_size=batch_size, validation_split=0.2,
          epochs=epochs, verbose=1, callbacks=[modelCheckpoint])


print(time.clock()-start)