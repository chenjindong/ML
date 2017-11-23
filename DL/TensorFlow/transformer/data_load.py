#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    '''
    Reutrns:
        word2inx: dict = {word: id}
        idx2word: dict = {id: word}
    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() 
                if int(line.split()[1])>=hp.min_cnt] # remove low term frequency words, threshold = 20
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    '''
    Returns:
        word2inx: dict = {word: id}
        idx2word: dict = {id: word}
    '''
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() 
                if int(line.split()[1])>=hp.min_cnt]  # remove low term frequency words, threshold = 20
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    '''
    get the word index list of sentence,for example
        'the cat is very cute' => [32,2564,7,32,873,0,0,0,0,0]
    Args:
        source_sents: English sentence list
        target_sents: German sentence list
    Returns:
        X: (n_Sources, 10), word index list of German sentence
        Y: (n_Targets, 10), word index list of English sentence 
        Sources: German sentence list
        Targets: English sentence list 
    '''
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index (mapping word to id,such as 'hello world' => [100,206])
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))  # [5108,274,...31,0,0,0]
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    return X, Y, Sources, Targets

def load_train_data():
    '''
    Returns
        X: (n_Sources, 10), word index list of German sentence
        Y: (n_Targets, 10), word index list of English sentence 
    '''
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n")                    if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n")                    if line and line[0] != "<"]
    
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y

def load_test_data():
    '''
    Returns
        X: (375, 10)
        Sources: (375) 
        Targets: (375)
    '''
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") 
                    if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") 
                    if line and line[:4] == "<seg"]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets 

def get_batch_data():
    '''
    get_batch_data
    :return x,y: (batch_size, 10)
    :return num_batch: data可以batch的数目
    '''
    # Load data
    X, Y = load_train_data()
    # calc total batch count
    num_batch = len(X) // hp.batch_size  #除法取整
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    return x, y, num_batch # (N, T), (N, T), ()

