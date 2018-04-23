import os, sys, h5py, ljqpy, jieba, math
import preprocess as pp

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import keras.losses
import keras.backend as K
import tensorflow as tf

from weight_norm import AdamWithWeightnorm
sen_input = Input(shape=(None,), dtype='int32')

sen = Lambda(lambda x:x[:,:-1])(sen_input)
target = Lambda(lambda x:x[:,1:])(sen_input)

emb_layer = Embedding(len(pp.id2c), 128)

x = emb_layer(sen)
x = GRU(128, return_sequences=True)(x)
x = GRU(128, return_sequences=True)(x)
x = TimeDistributed(Dense(len(pp.id2c), use_bias=False))(x)

def get_loss(args):
	y_pred, y_true = args
	y_true = tf.cast(y_true, 'int32')
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
	mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
	loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
	loss = K.mean(loss)
	return loss

def get_accu(args):
	y_pred, y_true = args
	mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
	corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
	corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
	return K.mean(corr)
	
loss = Lambda(get_loss)([x, target])
ppl = Lambda(K.exp)(loss)
accu = Lambda(get_accu)([x, target])

model = Model(sen_input, loss)
model.add_loss([K.mean(loss)])
model.compile(AdamWithWeightnorm(0.001), loss=None)
model.metrics_names.append('ppl')
model.metrics_tensors.append(ppl)
model.metrics_names.append('accu')
model.metrics_tensors.append(accu)

try: model.load_weights('data/lm.h5')
except: print('\n\nnew model\n')

def Ask(sen):
	x = [pp.c2id.get(z, 1) for z in sen][:pp.maxlen]
	xx = np.zeros((1, pp.maxlen), dtype='int32')
	xx[0,:len(x)] = x
	pred = model.predict_on_batch(xx)
	return pred
	sm = [z[w] for z, w in zip(pred, x[1:1+len(x)])]
	print(sm)
	sm = - np.log(np.array(sm) + 1e-10)
	return np.mean(sm)

if __name__ == '__main__':
	model.summary()
	with h5py.File('data/dataset.h5') as dfile: X = dfile['X'][:]
	model.fit(X, None, batch_size=512, epochs=3, \
		   validation_split=0.01,  \
		   callbacks=[ModelCheckpoint('data/lm.h5', save_best_only=True, save_weights_only=True),
				LearningRateScheduler(lambda x:0.1*(0.5**x))])
