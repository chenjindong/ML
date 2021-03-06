# -*- coding: utf-8 -*-
# @File    : seq_rnn.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/9 16:08

import tensorflow as tf


class SeqModelRNN(tf.keras.Model):
    def __init__(self, default_dim, seq_length, song_size, singer_size, sex_size, age_size, time_category_size,
                 rate_category_size, pre_embedding):
        super(SeqModelRNN, self).__init__()

        print("SeqModelRNN init")

        self.embedding_song = tf.keras.layers.Embedding(song_size, default_dim, trainable=False)
        self.embedding_song.build((None, song_size))
        self.embedding_song.set_weights([pre_embedding])

        self.embedding_singer = tf.keras.layers.Embedding(singer_size, default_dim)
        self.embedding_time_category = tf.keras.layers.Embedding(time_category_size, default_dim)
        self.embedding_rate = tf.keras.layers.Embedding(rate_category_size, default_dim)
        self.embedding_sex = tf.keras.layers.Embedding(sex_size, default_dim)
        self.embedding_age = tf.keras.layers.Embedding(age_size, default_dim)

        self.rnn_seq = tf.keras.layers.LSTM(default_dim)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(units=default_dim, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(units=default_dim, activation=tf.nn.selu)

        self.embedding_label = tf.keras.layers.Embedding(song_size, default_dim, trainable=False)
        self.embedding_label.build((None, song_size))
        self.embedding_label.set_weights([pre_embedding])

        self.dot = tf.keras.layers.Dot(axes=1, normalize=False)

        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs, training=False):
        predict_label = inputs["predict_label"]
        sex = inputs["sex"]
        age = inputs["age"]
        current_song = inputs["current_song"]
        current_time_category = inputs["current_time_category"]
        current_rate = inputs["current_rate"]
        current_singer = inputs["current_singer"]
        song = inputs["song"]
        time_category = inputs["time_category"]
        rate = inputs["rate"]
        singer = inputs["singer"]

        embedding_sex = self.embedding_sex(sex)
        embedding_age = self.embedding_age(age)

        embedding_current_song = self.embedding_song(current_song)
        embedding_current_singer = self.embedding_singer(current_singer)
        embedding_current_time_category = self.embedding_time_category(current_time_category)
        embedding_current_rate = self.embedding_rate(current_rate)

        embedding_history_song = self.embedding_song(song)
        embedding_history_singer = self.embedding_singer(singer)
        embedding_history_time_category = self.embedding_time_category(time_category)
        embedding_history_rate = self.embedding_rate(rate)

        embedding_current = tf.concat(
            [embedding_current_song, embedding_current_singer, embedding_current_time_category, embedding_current_rate],
            1)

        embedding_current = self.dense_1(embedding_current)

        embedding_history = tf.concat(
            [embedding_history_song, embedding_history_singer, embedding_history_time_category, embedding_history_rate],
            2)

        embedding_history_state = self.rnn_seq(embedding_history)

        x = tf.concat([embedding_sex, embedding_age, embedding_current, embedding_history_state], 1)

        x = self.dense_2(x)
        x = tf.math.l2_normalize(x, axis=1)

        if training:
            y = self.embedding_label(predict_label)
            y = self.flatten(y)
            y = tf.math.l2_normalize(y, axis=1)

            # z = tf.keras.backend.batch_dot(x, y)
            z = self.dot([x, y])
            z = self.output_layer(z)

            return z
        else:
            return x

        # return z, x
