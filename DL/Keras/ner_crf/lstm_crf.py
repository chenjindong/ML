import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

data = pd.read_csv("data/ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")

words = ['PAD'] + list(set(data["Word"].values))
# words.append("ENDPAD")
n_words = len(words)

tags = list(set(data["Tag"].values))
n_tags = len(tags)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sent = getter.get_next()  # [('Thousands', 'NNS', 'O'),....]

sentences = getter.sentences
max_len = 75
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)  # 固定句子的长度为75，小于75的补0对齐

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


y = [to_categorical(i, num_classes=n_tags) for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
y_tr, y_te = np.array(y_tr), np.array(y_te)
print('train: ', X_tr.shape, y_tr.shape)
print('test: ', X_te.shape, y_te.shape)

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words+1, output_dim=64, input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

model_checkpoint = ModelCheckpoint('lstm_crf_model.h5', save_best_only=True, save_weights_only=True)

model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=10,
          validation_split=0.1, verbose=1, callbacks=[model_checkpoint])

model.load_weights('lstm_crf_model.h5')
y_pred = model.predict(X_te)  # 3 dim

# print(y_te.shape)
y_te, y_pred = np.argmax(y_te, -1), np.argmax(y_pred, -1)
# print(y_te.shape)

report = flat_classification_report(y_pred=y_pred, y_true=y_te)
print(report)

i = 190
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], y_te[i], p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))


