import time
import sys

from databunch import train_file, test_file
from databunch import load_data, make_vocab, make_label, make_con_vocab, pre_trained_embedding
from databunch import maxSeqLen, maxConLen, vocab_size, con_size
from weight_norm import AdamWithWeightnorm

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping


import keras.backend as K

embedding_dim = 50
con_emb_dim = 50
batch_size = 64
epochs = 15

start = time.clock()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def CNNModel():
	# baseline 2: CNN
	print('run CNN model...')
	text_input = Input(shape=(maxSeqLen,), dtype='int32')

	# token embedding  , weights=[word_embeddings]
	text_emb = Embedding(vocab_size + 2, embedding_dim, trainable=True)(text_input)  # vocab_size+2: UNK,PAD

	text_emb = Conv1D(filters=50, kernel_size=3, padding='same')(text_emb)
	text_emb = GlobalMaxPooling1D()(text_emb)

	text_emb = Dense(30)(text_emb)

	output = Dense(10, activation='softmax')(text_emb)

	mm = Model(inputs=[text_input], outputs=output)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm
CNNModel()
def BilstmMaxpolingModel():
	# baseline3: BiLSTM + Maxpooling + MLP
	print('run BiLSTM + Maxpooling + MLP...')
	text_input = Input(shape=(maxSeqLen,), dtype='int32')

	# token embedding
	text_emb = Embedding(vocab_size + 2, embedding_dim, trainable=True, weights=[word_embeddings], mask_zero=True)(text_input)  # vocab_size+2: UNK,PAD
	text_emb = Masking()(text_emb)  # (30, 50)

	sent_emb = Bidirectional(LSTM(32, return_sequences=True))(text_emb)
	sent_emb = Lambda(lambda x: K.max(x, axis=1))(sent_emb)

	sent_emb = Dense(50)(sent_emb)
	output = Dense(class_num, activation='softmax')(sent_emb)

	mm = Model(inputs=text_input, outputs=output)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm

def FastTextModel():
	print('run CNN model...')
	text_input = Input(shape=(maxSeqLen,), dtype='int32')

	# token embedding
	text_emb = Embedding(vocab_size + 2, embedding_dim, weights=[word_embeddings], trainable=True)(text_input)  # vocab_size+2: UNK,PAD

	text_emb = Lambda(lambda x: K.sum(x, 1)/maxSeqLen)(text_emb)

	text_emb = Dense(30)(text_emb)

	output = Dense(class_num, activation='softmax')(text_emb)

	mm = Model(inputs=[text_input], outputs=output)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm

def SentEmbModel(multi=False):
	# baseline4: Sentence embedding (Bengio-ICLR 2017)
	print('run Sentence embedding (Bengio-ICLR 2017) model...')
	text_input = Input(shape=(maxSeqLen,), dtype='int32')

	# token embedding  , weights=[word_embeddings]
	text_emb = Embedding(vocab_size + 2, embedding_dim, mask_zero=True, trainable=True)(text_input)  # vocab_size+2: UNK,PAD
	# text_emb = SpatialDropout1D(0.5)(text_emb)  # drops entire 1D feature maps instead of individual elements
	text_emb = Masking()(text_emb)  # (30, 50)

	if not multi:
		# source2token self-attention
		lstm_output = Bidirectional(LSTM(32, return_sequences=True))(text_emb)
		da = 50
		alphalpha_prob = TimeDistributed(Dense(da, activation='tanh'))(lstm_output)
		alphalpha_prob = TimeDistributed(Dense(1, activation='softmax'))(alphalpha_prob)
		alphalpha_prob = Lambda(lambda x: x, output_shape=lambda s: s)(alphalpha_prob)
		alphalpha_prob = Reshape((maxSeqLen,))(alphalpha_prob)
		alphalpha_prob = RepeatVector(64)(alphalpha_prob)
		alphalpha_prob = Permute((2, 1))(alphalpha_prob)
		sent_emb = Multiply()([alphalpha_prob, lstm_output])  # alpha(i)*h(i)
		sent_emb = Lambda(lambda x: K.sum(x, axis=1), name='multiple')(sent_emb)
		sent_emb = Dense(50)(sent_emb)
	else:
		# source2token multi-dimentional self-attention
		lstm_output = Bidirectional(LSTM(32, return_sequences=True))(text_emb)
		da = 50
		alphalpha_prob = TimeDistributed(Dense(da, activation='tanh'))(lstm_output)
		r = 100
		alphalpha_prob = TimeDistributed(Dense(r, activation='softmax'))(alphalpha_prob)
		alphalpha_prob = Lambda(lambda x: x, output_shape=lambda s: s)(alphalpha_prob)
		alphalpha_prob = Permute((2, 1))(alphalpha_prob)
		sent_emb = Lambda(lambda x: K.batch_dot(x[0], x[1]))([alphalpha_prob, lstm_output])
		sent_emb = Lambda(lambda x: x, output_shape=lambda s: s)(sent_emb)
		sent_emb = Flatten()(sent_emb)
		sent_emb = Dense(50)(sent_emb)

	output = Dense(10, activation='softmax')(sent_emb)
	mm = Model(inputs=text_input, outputs=output)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm

SentEmbModel(True)
assert False
def WCCNNModel():
	# KPCNN (IJCAI 2017)
	print('run KPCNN model...')

	# token input
	text_input = Input(shape=(maxSeqLen,), dtype='int32')
	text_emb = Embedding(vocab_size + 2, embedding_dim, weights=[word_embeddings], trainable=True)(text_input)  # vocab_size+2: UNK,PAD
	# quest_emb = SpatialDropout1D(0.5)(quest_emb)
	text_emb = Conv1D(filters=50, kernel_size=3, padding='same')(text_emb)
	text_emb = GlobalMaxPooling1D()(text_emb)

	# concept input
	con_input = Input(shape=(maxConLen,), dtype='int32')
	con_emb = Embedding(con_size+1, con_emb_dim)(con_input)
	con_emb = Conv1D(filters=50, kernel_size=3, padding='same')(con_emb)
	con_emb = GlobalMaxPooling1D()(con_emb)

	concat = concatenate([text_emb, con_emb])
	final = Dense(50)(concat)
	final = Dense(class_num, activation='softmax')(final)

	mm = Model(inputs=[text_input, con_input], outputs=final)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm

def KPANNModel():

	text_input = Input(shape=(maxSeqLen,), dtype='int32')

	# CNN for sentence encoding 
	# text_emb = Embedding(vocab_size+2, embedding_dim, weights=[word_embeddings], trainable=True)(text_input)
	# sent_emb = Conv1D(filters=50, kernel_size=3, padding='same')(text_emb)
	# sent_emb = GlobalMaxPooling1D()(sent_emb)

	# LSTM for sentence encoding
	u=64
	text_emb = Embedding(vocab_size + 2, embedding_dim, mask_zero=True, weights=[word_embeddings], trainable=True)(text_input)  # vocab_size+2: UNK,PAD
	text_emb = Masking()(text_emb)  # (30, 50)
	sent_emb = Bidirectional(LSTM(u, return_sequences=True))(text_emb)
	sent_emb = Lambda(lambda x: K.max(x, axis=1))(sent_emb)

	# concept input
	con_input = Input(shape=(maxConLen,), dtype='int32')
	con_emb = Embedding(con_size + 1, con_emb_dim)(con_input)

	da = 70
	w1x = TimeDistributed(Dense(da))(con_emb)
	s2q = Dense(da)(sent_emb)
	s2q = RepeatVector(maxConLen)(s2q)
	mid_vec = Add()([s2q, w1x])
	mid_vec = Lambda(lambda x: K.tanh(x))(mid_vec)
	# mid_vec = Permute((2, 1))(mid_vec)
	alpha_prob = TimeDistributed(Dense(1))(mid_vec)
	alpha_prob = Flatten()(alpha_prob)
	alpha_prob = Softmax(name='alpha_prob')(alpha_prob)

	db = 35
	beta_prob = TimeDistributed(Dense(db, activation='tanh'))(con_emb)
	beta_prob = TimeDistributed(Dense(1))(beta_prob)
	beta_prob = Lambda(lambda x: x, output_shape=lambda s: s, name='mask')(beta_prob)
	beta_prob = Flatten()(beta_prob)
	beta_prob = Softmax(name='beta_prob')(beta_prob)
	gamma = 0.5
	atten_prob = Lambda(lambda x: gamma*x[0]+(1-gamma)*x[1], name='final_att')([alpha_prob, beta_prob])
	atten_prob = Softmax()(atten_prob)

	atten_prob = RepeatVector(con_emb_dim)(atten_prob)
	atten_prob = Permute((2, 1))(atten_prob)
	atte = Multiply()([atten_prob, con_emb])
	con_emb = Lambda(lambda x: K.sum(x, axis=1))(atte)

	concat = concatenate([sent_emb, con_emb])
	# concat = add([sent_emb, con_emb])
	final = Dense(50)(concat)
	final = Dense(class_num, activation='softmax')(final)

	mm = Model(inputs=[text_input, con_input], outputs=final)
	mm.summary()
	mm.compile(AdamWithWeightnorm(0.01), 'sparse_categorical_crossentropy', metrics=['accuracy'])
	return mm


def evaluate(y_true, y_pred):
	print(classification_report_imbalanced(y_true, y_pred))
	print('accuracy: ', accuracy_score(y_true, y_pred))


cmds = sys.argv[1:]

id2w, w2id = make_vocab()
id2con, con2id = make_con_vocab()
id2label, label2id = make_label(train_file)
word_embeddings = pre_trained_embedding()
X_train, y_train = load_data(train_file)
X_test, y_test = load_data(test_file)

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
class_num = len(set(label2id))
print('training data: ', X_train.shape, y_train.shape)
print('test data: ', X_test.shape, y_test.shape)
print('# of class: ', class_num)

model_saver = ModelCheckpoint('st_model.h5', save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

if 'prior' in cmds:
	# mm = WCCNNModel()
	mm = KPANNModel()
	mm.fit([X_train[:, :maxSeqLen], X_train[:, maxSeqLen:]], y_train,  batch_size=batch_size, epochs=epochs, verbose=2,
			   validation_split=0.15, callbacks=[model_saver, early_stopping])
	mm.load_weights('st_model.h5')

	y_pred = mm.predict([X_test[:, :maxSeqLen], X_test[:, maxSeqLen:]])
	y_pred = np.argmax(y_pred, 1)
	evaluate(y_test, y_pred)

	# get attention weight
	def attention_visuliaze():
		weight_lay = ['alpha_prob', 'beta_prob', 'final_att']
		inp = mm.input
		weights = {}
		for layer in mm.layers:
			if layer.name in weight_lay:
				func = K.Function(inp, [layer.output])
				weight = func([X_test[:, :maxSeqLen], X_test[:, maxSeqLen:]])[0]
				weights[layer.name] = weight
		for line in range(10000):
			sent, con = [], []
			for i in range(len(X_test[line])):
				if i>=maxSeqLen: con.append(id2con[int(X_test[line][i])])
				else: sent.append(id2w[int(X_test[line][i])])
			sent = [item for item in sent if item != '<PAD>']
			print(id2label[y_test[line]], id2label[y_pred[line]], ''.join(sent))
			print(con)
			print(weights['alpha_prob'][line], '||', (255-weights['alpha_prob'][line]*255))
			print(weights['beta_prob'][line], '||', (255-weights['beta_prob'][line]*255))
			print(weights['final_att'][line], '||', (255-weights['final_att'][line]*255))
			print()
else:
	# mm = CNNModel()
	mm = FastTextModel()
	# mm = BilstmMaxpolingModel()
	# mm = SentEmbModel()

	mm.fit(X_train[:, :maxSeqLen], y_train, batch_size=batch_size, epochs=epochs, verbose=2,
			validation_split=0.15, callbacks=[model_saver, early_stopping])

	mm.load_weights('st_model.h5')
	y_pred = mm.predict([X_test[:, :maxSeqLen], X_test[:, maxSeqLen:]])
	y_pred = np.argmax(y_pred, 1)
	

evaluate(y_test, y_pred)

print('running time: ', time.clock()-start)


