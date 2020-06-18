# -*- coding: utf-8 -*-
# @File    : build_data.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/14 17:25

import sys, codecs, random
from collections import defaultdict


def load_vocab(in_path):
    vocab_dict = defaultdict(dict)

    with codecs.open(in_path, "r", encoding="utf-8") as in_file:
        for line in in_file:
            cols = line.split("\t")
            vocab_dict[int(cols[0])][cols[1]] = int(cols[2])

    return vocab_dict


def get_vocab_size(vocab_dict, field_id):
    if field_id not in vocab_dict:
        return -1
    else:
        return len(vocab_dict[field_id])


def gen_pos_labels(labels, rates, start, end):
    pos_set = set()

    for i in range(start, end):
        if int(rates[i]) < 5:
            continue
        pos_set.add(labels[i])

    return pos_set


def neg_sampling(label_vocab_size, pos_labels, negative_size):
    neg_labels = list()

    for i in range(negative_size):
        neg_label = str(int(random.random() * label_vocab_size))
        if neg_label in pos_labels:
            continue
        neg_labels.append(neg_label)

    return neg_labels


def process(in_path, out_path, label_vocab_size, label_col_id, rate_col_id, seq_col_ids, point_col_ids, start_pos,
            window_train, window_predict, step_size, negative_size):
    line_num = 0

    with codecs.open(in_path, "r", encoding="utf-8") as in_file, \
            codecs.open(out_path, "w+", encoding="utf-8") as out_file:
        for line in in_file:
            line_num += 1
            if line_num % 1000000 == 0:
                print(line_num)

            try:
                cols = line.strip().split(",")

                labels = cols[label_col_id].split(" ")
                rates = cols[rate_col_id].split(" ")

                seq_cols = list()
                point_cols = list()

                for i, val in enumerate(seq_col_ids):
                    seq_cols.append(cols[val].split(" "))

                for i, val in enumerate(point_col_ids):
                    point_cols.append(cols[val].split(" ")[0])

                for i in range(start_pos, len(labels) - 1, step_size):
                    feature_list = list()

                    for point_col in point_cols:
                        feature_list.append(point_col)

                    feature_list.append(" ".join(labels[max(0, i - window_train):i]))

                    for seq_col in seq_cols:
                        feature_list.append(" ".join(seq_col[max(0, i - window_train):i]))

                    # pos_labels = set(labels[i:min(len(labels), i + window_predict)])
                    pos_labels = gen_pos_labels(labels, rates, i, min(len(labels), i + window_predict))
                    neg_labels = neg_sampling(label_vocab_size, pos_labels, negative_size)

                    out_file.write("%s,%s,%s\n" % (" ".join(pos_labels), " ".join(neg_labels), ",".join(feature_list)))
            except Exception as e:
                print(e)


if __name__ == "__main__":
    vocab = load_vocab(sys.argv[1])
    label_vocab_size = get_vocab_size(vocab, 1)
    label_col_id = 1
    rate_col_id = 2
    seq_col_ids = [0, 2, 5]
    point_col_ids = [3, 4]
    start_pos = 5
    window_train = 20
    window_predict = 5
    step_size = 5
    negative_size = 20
    process(sys.argv[2], sys.argv[3], label_vocab_size, label_col_id, rate_col_id, seq_col_ids, point_col_ids,
            start_pos, window_train, window_predict, step_size, negative_size)
