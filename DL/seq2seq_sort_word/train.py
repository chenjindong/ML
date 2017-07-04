#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8
'''
利用seq2seq对单词进行排序，比如common=>c m m n o o 
'''

import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import load_source_vocab, load_target_vocab, create_data, get_batches
from tensorflow.python.layers.core import Dense

graph = tf.Graph()

with graph.as_default():
    
    source_char2idx, source_idx2char = load_source_vocab()
    target_char2idx, target_idx2char = load_target_vocab()
   
    # (batch_size, valid sequence length)
    inputs = tf.placeholder(tf.int32, [None,None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    # 用于mask阶段
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_sequence_length')

    ############# Encode ###############
    # encode embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(inputs, len(source_char2idx), hp.embedding_size)
    
    # 将embedding 之后的向量encode_embed_input传给RNN
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        return lstm_cell

    encode_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hp.rnn_size) for _ in range(hp.num_layers)])
    # encoder_output: [batch_size, timesteps, rnn_size] 
    encoder_output, encoder_state = tf.nn.dynamic_rnn(encode_cell, 
                                                      encoder_embed_input, 
                                                      sequence_length=source_sequence_length,
                                                      dtype=tf.float32
                                                      )
    ################ Decoder ################
    # 将targets的最后一行<EOS>去掉,补上<GO>
    ending = tf.strided_slice(targets, [0, 0], [hp.batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([hp.batch_size, 1], target_char2idx['<GO>']), ending], 1) 
    
    # decode embedding
    #decoder_embed_input = tf.contrib.layers.embed_sequence(decoder_input, len(target_char2idx), hp.embedding_size)
    decoder_embeddings = tf.Variable(tf.random_uniform([len(target_char2idx), hp.embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
    
    # RNN cell in decoder 
    decode_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hp.rnn_size) for _ in range(hp.num_layers)])
    # output layer
    output_layer = Dense(len(target_char2idx),kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    
    # training decoder
    with tf.variable_scope('decode'):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False
                                                            )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer
                                                           )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length
                                                                          )
    # predicting decoder
    with tf.variable_scope('decode', reuse=True):
        start_tokens = tf.tile(tf.constant([target_char2idx['<GO>']], dtype=tf.int32), [hp.batch_size], name='start_tokens')
        
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, 
                                                                     start_tokens, 
                                                                     target_char2idx['<EOS>']
                                                                     )
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer
                                                             )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length # max time step
                                                                            )
    # (batch_size, timestep, embedding_size)
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    # (batch_size, timestep)
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope('optimization'):
            
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(hp.learning_rate) 

        # gradient clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

source_int,target_int = create_data()
train_source = source_int[hp.batch_size:]
train_target = target_int[hp.batch_size:]
#print(train_source[:5])
#print(train_target[:5])
valid_source = source_int[:hp.batch_size]
valid_target = target_int[:hp.batch_size]

valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths = next(get_batches(valid_source, valid_target))

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(hp.epochs):
        for sources_batch, targets_batch, sources_lengths, targets_lengths in get_batches(train_source, train_target):
#            print( [[source_idx2char[idx] for idx in word]  for word in sources_batch[:5]])
#            print( [[target_idx2char[idx] for idx in word]  for word in targets_batch[:5]])
#            assert False
            _, loss = sess.run([train_op, cost], {inputs: sources_batch, 
                                                  targets: targets_batch, 
                                                  source_sequence_length: sources_lengths,
                                                  target_sequence_length: targets_lengths
                                                 })
        validation_loss = sess.run(cost, {inputs: valid_sources_batch,
                                            targets: valid_targets_batch,
                                            source_sequence_length: valid_sources_lengths,
                                            target_sequence_length: valid_targets_lengths
                                            })
        print('epcho: %d, training loss: %f, valid loss: %f' % (epoch_i, loss, validation_loss))
    
    saver = tf.train.Saver()
    saver.save(sess, hp.checkpoint)
    print('Model trained and saved')













