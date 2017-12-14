import time
import ljqpy
import numpy as np
from databunch import maxQLen, maxWordLen, vocab_size, char_size
from databunch import CutSentence, Tokens2Intlist, Chars2Intlist, MakeVocab


from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dense, SpatialDropout1D, Masking, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.wrappers import TimeDistributed

import keras.backend as K
import tensorflow as tf


'''
task:
    input：question
    output：answer type
example
    input：谁演唱了青花瓷？
    output：人物
learned:
    1. 重写GlobalAveragePooling1D
    2. SpatialDropout1D
    3. BatchNormalization
    4. Masking
    5. TimeDistributed
    6. 重写Adam
    7. ModelCheckpoint
'''

time.clock()

embedding_dim = 16
char_embd_dim = 16
batch_size = 200
epochs = 50
trainFile = r'data/QATdata.txt'
validateFile = r'data/QATtest.txt'

class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())  # mask (batch, time)
            mask = K.repeat(mask, x.shape[-1])  # mask (batch, x_dim, time)
            mask = tf.transpose(mask, [0, 2, 1])  # mask (batch, time, x_dim)
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

def model():
    quest_input = Input(shape=(maxQLen,), dtype='int32')
    questC_input = Input(shape=(maxQLen, maxWordLen), dtype='int32')

    # token embedding
    embed_layer = Embedding(vocab_size + 2, embedding_dim, mask_zero=True)
    quest_emb = SpatialDropout1D(0.8)(embed_layer(quest_input))
    quest_emb = BatchNormalization()(quest_emb)
    quest_emb = Masking()(quest_emb)  # (None, 16)

    # char embedding
    char_input = Input(shape=(maxWordLen,), dtype='int32')  # (None, 5)
    cembed_layer = Embedding(char_size + 2, char_embd_dim)(char_input)
    c_emb = SpatialDropout1D(0.8)(cembed_layer)
    cc = BatchNormalization()(c_emb)
    cc = Conv1D(char_embd_dim, 3, padding='same')(cc)
    cc = LeakyReLU()(cc)
    cc = Lambda(lambda x: K.sum(x, 1), output_shape=lambda d: (d[0], d[2]))(cc)  # (None, 16)
    char_model = Model(inputs=char_input, outputs=cc)
    char_model.summary()

    qc_emb = TimeDistributed(char_model)(questC_input)

    quest_emb = concatenate([quest_emb, qc_emb])  # (None, 30, 32)

    final = GlobalAveragePooling1DMasked()(quest_emb)  # Polling1D只能处理2阶tensor
    final = Dense(typelen, activation='softmax')(final)

    from weight_norm import AdamWithWeightnorm

    mm = Model(inputs=[quest_input, questC_input], outputs=final)
    mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return mm

def MakeOwnDatas(train):
    xQuestion = []
    xQuestionC = []
    for line in ljqpy.LoadCSV(train):
        question = line[0]
        questionTokens = CutSentence(question)
        xq = Tokens2Intlist(questionTokens, maxQLen)
        xqc = Chars2Intlist(questionTokens, maxQLen)
        xQuestion.append(xq)
        xQuestionC.append(xqc)
    return np.array(xQuestion), np.array(xQuestionC)

def makeType(train, valid):
    global typelen
    global type2num
    answertypelist = []
    trainl = ljqpy.LoadCSV(train)
    validl = ljqpy.LoadCSV(valid)
    for each in trainl:
        answertype = each[2]
        answertypelist.append(answertype)
    for each2 in validl:
        answertype2 = each2[2]
        answertypelist.append(answertype2)
    answertypelist = list(set(answertypelist))
    typelen = len(answertypelist)
    type2num = {y: x for x, y in enumerate(answertypelist)}
    qy = []
    for each in trainl:
        t = each[2]
        qy.append(type2num[t])
    tyq = []
    for each in validl:
        t = each[2]
        tyq.append(type2num[t])
    return np.array(qy), np.array(tyq)

if __name__ == '__main__':

    MakeVocab()
    Xq, Xqc = MakeOwnDatas(trainFile)
    tXq, tXqc = MakeOwnDatas(validateFile)
    yq, tyq = makeType(trainFile, validateFile)
    print(Xq.shape, Xqc.shape, yq.shape)
    print(tXq.shape, tXqc.shape, tyq.shape)
    mm = model()
    mm.summary()
    print('load ok %.3f' % time.clock())
    mm.fit([Xq, Xqc], yq, batch_size=batch_size, epochs=epochs, verbose=2,
            validation_data=([tXq, tXqc], tyq),
            callbacks=[ModelCheckpoint('anstype.h5', save_weights_only=False, save_best_only=True, period=5)])

    print('completed')