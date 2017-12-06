#!/home/user0/anaconda3/bin/python
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.engine.topology import Input
from keras.layers.merge import Dot
from keras.layers.core import Reshape, Dense
from keras.preprocessing.sequence import skipgrams
import numpy as np

'''
Using keras to implement skip-gram algorithm
'''

embedding_dims = 300
batch_size = 32
epochs = 5
path = r'\\10.141.208.22\data\Chinese_isA\corpus\wikicorpus_seg.txt'

# dataset
texts = []
vocabulary = []
with open(path, 'r', encoding='utf-8') as fin:
    for i, line in enumerate(fin):
        if i > 10000:
            break
        tokens = line.strip().split('\t')
        vocabulary.extend(tokens)
        texts.append(tokens)
vocabulary = set(vocabulary)
vocab_size = len(vocabulary)
word2id, id2word = {}, {}
for idx, word in enumerate(vocabulary):
    word2id[word] = idx
    id2word[idx] = word
# text_seq = []
# for text in texts:
#     text_seq.append([word2id[word] for word in text])
text_seq = [word2id[word] for text in texts for word in text]
pairs, labels = skipgrams(text_seq, len(word2id), window_size=4)
x_train, y_train = pairs[:-10000], labels[:-10000]
x_test, y_test = pairs[-10000:], labels[-10000:]
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# model
input1 = Input(shape=(1,))
embeddings1 = Embedding(vocab_size, embedding_dims)(input1)
output1 = Reshape(target_shape=(embedding_dims,))(embeddings1)

input2 = Input(shape=(1,))
embeddings2 = Embedding(vocab_size, embedding_dims)(input2)
output2 = Reshape(target_shape=(embedding_dims,))(embeddings2)

dot = Dot(axes=-1)([output1, output2])
output = Dense(units=1, activation='sigmoid')(dot)
model = Model(inputs=(input1, input2), outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(x=[x_train[:, 0], x_train[:, 1]], y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)


