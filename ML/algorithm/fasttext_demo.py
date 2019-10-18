#!/bin/python3
import fasttext

import cjdpy
import random
import json
from collections import Counter

# https://fasttext.cc/docs/en/python-module.html

def get_feature(sent):
    # get 1-gram and 2-gram feature from sentences
    chars = []
    for char in sent:
        if '\u4e00' <= char <= '\u9fff':  # u is necessary
            chars.append(char)
    ngram = []
    for j in range(len(chars)-1):
        ngram.append(chars[j])
        ngram.append(chars[j] + chars[j+1])
    fname = ' '.join(ngram)
    return fname

import time
start = time.time()
data = cjdpy.load_csv("data/original_data")
random.shuffle(data)
res = []
for i in range(len(data)):
    res.append([get_feature(data[i][1]), "__label__"+data[i][0]])


# data format
# 毛 毛毛 毛 毛虫 虫 虫的 的 的意 意 意见    __label__1
cjdpy.save_csv(res[:300000], "data/train.txt")
cjdpy.save_csv(res[300000:], "data/test.txt")
classifier = fasttext.train_supervised("data/train.txt")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
            
print_results(*classifier.test('data/test'))

pred = classifier.predict(pred_list)
print(time.time()-start)
