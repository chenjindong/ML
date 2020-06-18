#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : build_vocab_mapper.py.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/23 20:46

import sys


def mapper(col_ids):
    for line in sys.stdin:
        try:
            cols = line.strip().split("\t")[1].split("\1")

            for col_id in col_ids:
                print("%s\t%s" % (str(col_id), cols[col_id]))
        except:
            pass


if __name__ == "__main__":
    vocab_col_ids = [1, 5, 11, 12]
    mapper(vocab_col_ids)
