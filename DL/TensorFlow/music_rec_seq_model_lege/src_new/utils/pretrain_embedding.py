# -*- coding: utf-8 -*-
# @File    : build_data.py
# @Author  : letian
# @Mail    : @.com

import codecs
import numpy as np


def load_embedding(file_path, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    i = 0

    with codecs.open(file_path, encoding='utf8') as in_file:
        for line in in_file:
            values = line.strip().split()
            vec = np.asarray(values, dtype='float32')

            if i < vocab_size:
                embedding_matrix[i] = vec
                i += 1
            else:
                print("embedding overflow")
                return None

    return embedding_matrix
