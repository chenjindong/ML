#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : build_vocab_reducer.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/23 20:47

import sys
from collections import defaultdict


def reducer():
    vocab_dict = defaultdict(int)

    for line in sys.stdin:
        tokens = line.strip("\n").split("\t")

        if len(tokens) != 2:
            continue

        vol_id = tokens[0]
        words = tokens[1].split(";")

        for word in words:
            key = "%s\t%s" % (vol_id, word)
            vocab_dict[key] += 1

    count_list = sorted(vocab_dict.iteritems(), key=lambda x: x[1], reverse=True)

    for k, v in count_list:
        print("%s\t%s" % (k, v))


if __name__ == "__main__":
    reducer()
