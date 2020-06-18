# -*- coding: utf-8 -*-
# @File    : tf_record_format.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/19 11:53

import sys, codecs, random
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
import multiprocessing as mp

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 20, 'Sequence length.')
flags.DEFINE_integer('pos_length', 500, 'Sequence length.')
flags.DEFINE_integer('neg_length', 20, 'Sequence length.')
flags.DEFINE_string('in_path', "in.txt", 'Sequence length.')
flags.DEFINE_string('out_path', "out.record", 'Sequence length.')
flags.DEFINE_integer('thread_num', 6, 'Thread num.')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature_list(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_one_line(line):
    cols = line.split(",")

    # pos_label = int(random.choice(cols[0].split(" ")))
    # neg_label = int(random.choice(cols[1].split(" ")))
    pos_label = [int(x) for x in cols[0].split(" ")]
    neg_label = [int(x) for x in cols[1].split(" ")]

    if len(pos_label) < 1 or len(neg_label) < 1:
        raise Exception(len(pos_label), len(neg_label))

    sex = int(cols[2])
    age = int(cols[3])
    song = [int(token) for token in cols[4].split(" ")]
    time_category = [int(token) for token in cols[5].split(" ")]
    rate = [int(token) for token in cols[6].split(" ")]
    singer = [int(token) for token in cols[7].split(" ")]
    current_song = song[-1]
    current_time_category = time_category[-1]
    current_rate = rate[-1]
    current_singer = singer[-1]

    pos_label.extend([0] * (FLAGS.pos_length - len(pos_label)))
    neg_label.extend([0] * (FLAGS.neg_length - len(neg_label)))

    song.extend([0] * (FLAGS.seq_length - len(song)))
    time_category.extend([0] * (FLAGS.seq_length - len(time_category)))
    rate.extend([0] * (FLAGS.seq_length - len(rate)))
    singer.extend([0] * (FLAGS.seq_length - len(singer)))

    # binary_label = 1
    # predict_label = pos_label

    # if random.random() > 0.33:
    #    binary_label = 0
    #    predict_label = neg_label

    feature = {
        # 'binary_label': _int64_feature(np.int32(binary_label)),
        # 'predict_label': _int64_feature(np.int32(predict_label)),
        'pos_label': _int64_feature_list(np.int32(pos_label)),
        'neg_label': _int64_feature_list(np.int32(neg_label)),
        'sex': _int64_feature(np.int32(sex)),
        'age': _int64_feature(np.int32(age)),
        'current_song': _int64_feature(np.int32(current_song)),
        'current_time_category': _int64_feature(np.int32(current_time_category)),
        'current_rate': _int64_feature(np.int32(current_rate)),
        'current_singer': _int64_feature(np.int32(current_singer)),
        'song': _int64_feature_list(np.int32(song)),
        'time_category': _int64_feature_list(np.int32(time_category)),
        'rate': _int64_feature_list(np.int32(rate)),
        'singer': _int64_feature_list(np.int32(singer))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def process(argv):
    with tf.io.TFRecordWriter(FLAGS.out_path) as writer:
        with codecs.open(FLAGS.in_path, "r", encoding="utf-8") as in_file:
            for line in in_file:
                record = process_one_line(line)
                writer.write(record.SerializeToString())


def process_worker(thread_num, thread_id):
    with tf.io.TFRecordWriter(FLAGS.out_path + "_" + str(thread_id)) as writer:
        with codecs.open(FLAGS.in_path, "r", encoding="utf-8") as in_file:
            id = 0
            for line in in_file:
                try:
                    id += 1
                    if id % thread_num != thread_id:
                        continue
                    record = process_one_line(line)
                    writer.write(record.SerializeToString())
                except Exception as e:
                    pass


def multi_process(argv):
    pool = mp.Pool(FLAGS.thread_num)
    jobs = []

    for i in range(FLAGS.thread_num):
        jobs.append(pool.apply_async(process_worker, (FLAGS.thread_num, i)))

    for job in jobs:
        job.get()

    pool.close()


if __name__ == "__main__":
    app.run(multi_process)
