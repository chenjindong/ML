#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : build_data_mapper.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/17 10:38


import sys, random


def mapper():
    for i, line in enumerate(sys.stdin):
        if random.random() < 0.005:
            print(line)


if __name__ == "__main__":
    mapper()
