#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : filter_data_mapper.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/23 21:21

import sys


def mapper():
    for line in sys.stdin:
        try:
            cols = line.strip().split("\t")[1].split("\1")
            seq_len = len(cols[1].split(";"))

            if seq_len < 50000 and seq_len >= 5:
                print(line.strip("\n"))
        except:
            pass


if __name__ == "__main__":
    mapper()
