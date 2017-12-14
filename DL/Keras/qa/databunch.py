import jieba, time
import os, ljqpy
import numpy as np
import re

time.clock()

MODE = 'train'

maxQLen = 30
maxWordLen = 5

vocab_size = 4500
char_size = 3000

batch_size = 200

def CutSentence(s):
    return jieba.lcut(s)

'''
#wordlist: 汉语短语对应 标号
#charlist: 字典

'''
def MakeVocab():
    global id2w, w2id, id2c, c2id
    vocabFile = 'data/wordlist.txt'
    charFile = 'data/charlist.txt'
    freqw = []
    freqc = []
    if os.path.exists(vocabFile):
        freqw = ljqpy.LoadCSV(vocabFile)
        freqc = ljqpy.LoadCSV(charFile)
    else:
        print('wordlist or charlist is not find')
    id2w = ['<PAD>', '<UNK>'] + [x[0] for x in freqw[:vocab_size]]
    w2id = {y: x for x, y in enumerate(id2w)}
    id2c = ['<PAD>', '<UNK>'] + [x[0] for x in freqc[:char_size]]
    c2id = {y: x for x, y in enumerate(id2c)}


def ChangeToken(token):
    if token.isdigit():
        token = '<NUM%d>' % min(len(token), 6)
    elif re.match('^[a-zA-Z]+$', token):
        token = '<ENG>'
    return token


def Tokens2Intlist(tokens, maxSeqLen):
    '''token数值化'''
    ret = np.zeros(maxSeqLen)
    tokens = tokens[:maxSeqLen]
    for i, t in enumerate(tokens):
        t = ChangeToken(t)
        ret[maxSeqLen - len(tokens) + i] = w2id.get(t, 1)
    return ret


def Chars2Intlist(tokens, maxSeqLen):
    '''字数值化'''
    ret = np.zeros((maxSeqLen, maxWordLen))
    tokens = tokens[:maxSeqLen]
    for i, t in enumerate(tokens):
        for j, c in enumerate(t[:maxWordLen]):
            ret[maxSeqLen - len(tokens) + i, j] = c2id.get(c, 1)
    return ret

