# -*- coding: utf-8 -*-
# @File    : format_dataset.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/13 17:39


import sys, codecs, datetime
from collections import defaultdict


def load_vocab(in_path):
    vocab_dict = defaultdict(dict)

    with codecs.open(in_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            cols = line.split("\t")
            vocab_dict[int(cols[0])][cols[1]] = int(cols[2])

    return vocab_dict


def vocab_lookup(vocab_dict, filed_id, key):
    if filed_id not in vocab_dict:
        return 0
    if key not in vocab_dict[filed_id]:
        return 0
    return vocab_dict[filed_id][key]


def time_category(timestamp):
    try:
        dt = datetime.datetime.fromtimestamp(float(timestamp))
        return int(dt.hour / 3)
    except Exception as e:
        print(e)
        return 0


def age_category(age):
    try:
        return int(int(age) / 5)
    except Exception as e:
        return 0


def float_format(rate):
    try:
        return round(float(rate), 2)
    except Exception as e:
        print(e)
        return rate


def format_process(vocab_dict, timestamp_col_ids, age_col_ids, float_col_ids, delete_col_ids, in_path, out_path):
    line_num = 0

    with codecs.open(in_path, "r", encoding="utf-8") as in_file, \
            codecs.open(out_path, "w+", encoding="utf-8") as out_file:
        for line in in_file:
            line_num += 1
            if line_num % 1000000 == 0:
                print(line_num)

            try:
                cols = line.strip().split("\t")[1].split("\1")

                for filed_id in timestamp_col_ids:
                    tokens = cols[filed_id].split(";")
                    token_ids = [str(time_category(token)) for token in tokens]
                    cols[filed_id] = " ".join(token_ids)

                for filed_id in age_col_ids:
                    tokens = cols[filed_id].split(";")
                    token_ids = [str(age_category(token)) for token in tokens]
                    cols[filed_id] = " ".join(token_ids)

                for filed_id in float_col_ids:
                    tokens = cols[filed_id].split(";")
                    token_ids = [str(max(0, min(int(float_format(token) * 10), 10))) for token in tokens]
                    cols[filed_id] = " ".join(token_ids)

                for filed_id in vocab_dict.keys():
                    tokens = cols[filed_id].split(";")
                    token_ids = [str(vocab_lookup(vocab_dict, filed_id, token)) for token in tokens]
                    cols[filed_id] = " ".join(token_ids)

                cols_final = list()

                for i, val in enumerate(cols):
                    if i not in delete_col_ids:
                        cols_final.append(val)

                out_file.write(",".join(cols_final) + "\n")
            except:
                print("error line")


if __name__ == "__main__":
    timestamp_col_ids = [0]
    age_col_ids = [6]
    float_col_ids = [4]
    delete_col_ids = {2, 3, 7, 8, 9, 10, 11}
    vocab_dict = load_vocab(sys.argv[1])
    format_process(vocab_dict, timestamp_col_ids, age_col_ids, float_col_ids, delete_col_ids, sys.argv[2], sys.argv[3])
