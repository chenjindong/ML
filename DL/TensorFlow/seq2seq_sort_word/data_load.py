#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8

from hyperparams import Hyperparams as hp
import numpy as np

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')
        special_tag = ['<UNK>', '<PAD>', '<GO>', '<EOS>']
        vocab = list(sorted(set([ch for word in words for ch in word])))
        char2idx = {char: idx for idx, char in enumerate(special_tag + vocab)}
        idx2char = {idx: char for idx, char in enumerate(special_tag + vocab)}
        return char2idx, idx2char

def load_source_vocab():
    return load_vocab(hp.source_data_path)

def load_target_vocab():
    return load_vocab(hp.target_data_path)

def create_data():
    '''
    数字化(bsaqq => [23, 9, 15, 8, 8])
    Returns:
        source_int: [[23, 9, 15, 8, 8], [5, 24, 18], [2, 23, 6, 3, 22],...] 
        target_int: [[23, 1, 12, 12, 13, 3], [18, 3, 8, 3], [1, 16, 6, 22, 4, 3],...] 
    '''
    source_char2idx, _ = load_source_vocab()
    target_char2idx, _ = load_target_vocab()
    source_int = []
    target_int = []
    with open(hp.source_data_path, 'r', encoding='utf-8') as fsource:
        words = fsource.read().split('\n')
        for word in words:
            word_int = [source_char2idx.get(char, 0) for char in word] # 0 means <UNK>
            source_int.append(word_int)
    with open(hp.target_data_path, 'r', encoding='utf-8') as ftarget:
        words = ftarget.read().split('\n')
        for word in words:
            word_int = [target_char2idx.get(char, 0) for char in word]
            word_int.append(3)  # 3 means <EOS>
            target_int.append(word_int)
    return source_int, target_int

def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int]*((max_sentence)-len(sentence)) for sentence in sentence_batch]

def get_batches(sources, targets):
    
    for i in range(0, len(sources)//hp.batch_size):
        start = i * hp.batch_size
        sources_batch = sources[start: start + hp.batch_size]
        targets_batch = targets[start: start + hp.batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, 1))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, 1))

        valid_sources_lengths = np.array([len(source) for source in sources_batch])
        valid_targets_lengths = np.array([len(target) for target in targets_batch])
        '''
        pad_sources_batch = []
        for source in sources_batch:
            pad_sources_batch.append(source + (max(valid_sources_lengths)-len(source))*[1]) # 1 means <PAD>
        pad_targets_batch = []
        for target in targets_batch:
            pad_targets_batch.append(target + (max(valid_targets_lengths)-len(target))*[1])
        '''
        yield pad_sources_batch, pad_targets_batch, valid_sources_lengths,valid_targets_lengths
        #yield pad_sources_batch, pad_targets_batch, [max(valid_sources_lengths)]*hp.batch_size, [max(valid_targets_lengths)]*hp.batch_size
