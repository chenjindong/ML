# -*- coding: utf-8 -*-
# @File    : build_vocab.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/9 16:09

import sys, codecs
from collections import defaultdict


def build_vocab(in_path, col_ids):
    vocab_dict = defaultdict(set)

    line_num = 0

    with codecs.open(in_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            line_num += 1
            if line_num % 1000000 == 0:
                print(line_num)

            try:
                cols = line.strip().split("\t")[1].split("\1")

                for col_id in col_ids:
                    tokens = cols[col_id].split(";")
                    vocab_dict[col_id].update(tokens)
            except:
                print("error line")

    return vocab_dict


def dump_dict(out_path, vocab_dict):
    with codecs.open(out_path, "w+", encoding="utf-8") as out_file:
        for k, v_set in vocab_dict.items():
            id = 1
            for v in v_set:
                out_file.write("%s\t%s\t%d\n" % (k, v, id))
                id += 1


if __name__ == "__main__":
    vocab_col_ids = [1, 5, 11, 12]
    vocab_dict = build_vocab(sys.argv[1], vocab_col_ids)
    dump_dict(sys.argv[2], vocab_dict)
