#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : format_dataset_mapper.py.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/12/16 22:53

import sys


def mapper():
    for i, line in enumerate(sys.stdin):
        print(line)


if __name__ == "__main__":
    mapper()
