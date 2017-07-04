#!/home/gdmlab/anaconda3/bin/python
#coding: utf-8

import tensorflow as tf
from data_load import load_source_vocab, load_target_vocab
from hyperparams import Hyperparams as hp

source_char2idx, source_idx2char = load_source_vocab()
target_char2idx, target_idx2char = load_target_vocab()

word = 'commonfdasyuiwenkda'

word_int = [source_char2idx.get(char, 0) for char in word] # 0 means <UNK>

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    loader = tf.train.import_meta_graph(hp.checkpoint + '.meta')
    loader.restore(sess, hp.checkpoint)
    
    inputs = graph.get_tensor_by_name('inputs:0')
    logits = graph.get_tensor_by_name('predictions:0')
    source_sequence_length = graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = graph.get_tensor_by_name('target_sequence_length:0')
    
    answer_logits = sess.run(logits, {inputs: [word_int]*hp.batch_size,
                                      source_sequence_length: [len(word_int)]*hp.batch_size,
                                      target_sequence_length: [len(word_int)]*hp.batch_size})[0]

print('source: ', word)
print('source index: ', word_int)
print('target index:', answer_logits)
res = [target_idx2char.get(i,0) for i in answer_logits]
print('target: ', res)







