from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf

from dataloader import TokenList, pad_to_longest
'''
keras实现seq2seq
将十进制数转化成十六进制
'''
class Encoder():
	def __init__(self, i_token_num, latent_dim, layers=3):
		self.emb_layer = Embedding(i_token_num, latent_dim, mask_zero=True)
		cells = [GRUCell(latent_dim) for _ in range(layers)]
		self.rnn_layer = RNN(cells, return_state=True)  # return_state
	def __call__(self, x):
		x = self.emb_layer(x)
		xh = self.rnn_layer(x)
		x, h = xh[0], xh[1:]
		return x, h

class Decoder():
	def __init__(self, o_token_num, latent_dim, layers=3):
		self.emb_layer = Embedding(o_token_num, latent_dim, mask_zero=True)
		cells = [GRUCell(latent_dim) for _ in range(layers)]
		self.rnn_layer = RNN(cells, return_sequences=True, return_state=True)  # return_sequences, return_state
		self.out_layer = Dense(o_token_num)
	def __call__(self, x, state):
		x = self.emb_layer(x)
		xh = self.rnn_layer(x, initial_state=state)  # initial_state
		x, h = xh[0], xh[1:]
		x = TimeDistributed(self.out_layer)(x)
		return x, h

class RNNSeq2Seq:
	def __init__(self, i_tokens, o_tokens, latent_dim, layers=3):
		self.i_tokens = i_tokens
		self.o_tokens = o_tokens

		encoder_inputs = Input(shape=(None,), dtype='int32')  # input sequence layer
		decoder_inputs = Input(shape=(None,), dtype='int32')  # output sequence layer

		encoder = Encoder(i_tokens.num(), latent_dim, layers)
		decoder = Decoder(o_tokens.num(), latent_dim, layers)
		
		encoder_outputs, encoder_states = encoder(encoder_inputs)  # encode阶段，encoder_states相当于input sequence的语义信息

		dinputs = Lambda(lambda x:x[:,:-1])(decoder_inputs)  # decoder input 很关键
		dtargets = Lambda(lambda x:x[:,1:])(decoder_inputs)  # derocer target

		decoder_outputs, decoder_state_h = decoder(dinputs, encoder_states)  # decode阶段

		def get_loss(args):
			'''user defined loss: minimize the difference between target sequence and output sequence'''
			y_pred, y_true = args
			y_true = tf.cast(y_true, 'int32')
			# 注意sparse_softmax_cross_entropy_with_logits和softmax_cross_entropy_with_logits
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
			mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
			loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
			loss = K.mean(loss)
			return loss

		def get_accu(args):
			y_pred, y_true = args
			mask = K.cast(tf.not_equal(y_true, 0), 'float32')
			corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
			corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
			return K.mean(corr)
				
		loss = Lambda(get_loss)([decoder_outputs, dtargets])  # 自定义loss的第二种方式：将loss作为model的output层

		# 自定义的另外两个metric, 类似于loss
		self.ppl = Lambda(K.exp)(loss)  # ppl越小，说明输出序列和目标序列越相近
		self.accu = Lambda(get_accu)([decoder_outputs, dtargets])

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=loss)  # 第二种加loss的方式
		self.model.add_loss([K.mean(loss)])
		

		# 以下代码和预测有关
		encoder_model = Model(encoder_inputs, encoder_states)

		decoder_states_inputs = [Input(shape=(latent_dim,)) for _ in range(layers)]

		decoder_outputs, decoder_states = decoder(decoder_inputs, decoder_states_inputs)
		decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
							  outputs=[decoder_outputs] + decoder_states)
		self.encoder_model = encoder_model
		self.decoder_model = decoder_model

	def compile(self, optimizer):
		self.model.compile(optimizer, loss=None)
		self.model.metrics_names.append('ppl')
		self.model.metrics_tensors.append(self.ppl)
		self.model.metrics_names.append('accu')
		self.model.metrics_tensors.append(self.accu)

	def decode_sequence(self, input_seq, delimiter=''):
		
		# input sequence processing
		input_mat = np.zeros((1, len(input_seq)+3))
		input_mat[0,0] = self.i_tokens.id('<S>')
		for i, z in enumerate(input_seq): input_mat[0,1+i] = self.i_tokens.id(z)
		input_mat[0,len(input_seq)+1] = self.i_tokens.id('</S>')

		state_value = self.encoder_model.predict_on_batch(input_mat)  # each layer has a hidden state
		target_seq = np.zeros((1, 1))
		target_seq[0,0] = self.o_tokens.id('<S>')

		decoded_tokens = []
		while True:
			output_tokens_and_h = self.decoder_model.predict_on_batch([target_seq] + state_value)
			output_tokens, h = output_tokens_and_h[0], output_tokens_and_h[1:]
			sampled_token_index = np.argmax(output_tokens[0,-1,:])
			sampled_token = self.o_tokens.token(sampled_token_index)
			decoded_tokens.append(sampled_token)
			if sampled_token == '</S>' or len(decoded_tokens) > 50: break
			target_seq = np.zeros((1, 1))
			target_seq[0,0] = sampled_token_index
			state_value = h

		return delimiter.join(decoded_tokens[:-1])


import random
import numpy as np

if __name__ == '__main__':
	itokens = TokenList(list('0123456789'))
	otokens = TokenList(list('0123456789abcdefx'))

	def GenSample():
		# 随机生成一个十进制和它的十六进制数字
		x = random.randint(0, 99999)
		y = hex(x); x = str(x)
		return x, y

	X, Y = [], []
	for _ in range(100000):
		x, y = GenSample()
		X.append(list(x))
		Y.append(list(y))

	X, Y = pad_to_longest(X, itokens), pad_to_longest(Y, otokens)
	print('input shape: ', X.shape)
	print('output shape: ', Y.shape)

	s2s = RNNSeq2Seq(itokens, otokens, 128, layers=2)
	s2s.compile('rmsprop')
	s2s.model.summary(line_length=120, positions=[.20, .55, .67, 1.])
	# assert False

	class TestCallback(Callback):
		def on_epoch_end(self, epoch, logs = None):
			print('\n')
			for test in [123, 12345, 34567]:
				ret = s2s.decode_sequence(str(test))
				print(test, ret, hex(test))
			print('\n')

	#s2s.model.load_weights('model.h5')
	s2s.model.fit([X, Y], None, batch_size=256, epochs=30, verbose=1,
					 validation_split=0.05, callbacks=[TestCallback()])

	s2s.model.save_weights('seq2seq_model.h5')


