# -*- coding: utf-8 -*-
# @File    : seq_rnn.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/9 16:08

import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dense = tf.keras.layers.Dense(d_model, activation=tf.nn.relu)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = self.dense(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class SeqModelTransformerSoftmax(tf.keras.Model):
    def __init__(self, default_dim, seq_length, song_size, singer_size, sex_size, age_size, time_category_size,
                 rate_category_size, pre_embedding):
        super(SeqModelTransformerSoftmax, self).__init__()

        print("SeqModelTransformerSoftmax init")

        self.embedding_song = tf.keras.layers.Embedding(song_size, default_dim, weights=[pre_embedding],
                                                        trainable=False)
        self.embedding_singer = tf.keras.layers.Embedding(singer_size, default_dim)
        self.embedding_time_category = tf.keras.layers.Embedding(time_category_size, default_dim)
        self.embedding_rate = tf.keras.layers.Embedding(rate_category_size, default_dim)
        self.embedding_sex = tf.keras.layers.Embedding(sex_size, default_dim)
        self.embedding_age = tf.keras.layers.Embedding(age_size, default_dim)

        self.transformer_encoder = Encoder(num_layers=3, d_model=64, num_heads=8, dff=default_dim,
                                           maximum_position_encoding=20)

        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(units=default_dim, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(units=default_dim, activation=tf.nn.selu)

        pre_embedding_reshape = pre_embedding.T

        self.output_layer = tf.keras.layers.Dense(units=song_size, activation='softmax', use_bias=False,
                                                  trainable=False)
        self.output_layer.build((None, default_dim))
        self.output_layer.set_weights([pre_embedding_reshape])

    @tf.function
    def call(self, inputs, training=False):
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

        transformer_encoder = self.transformer_encoder(embedding_history, training, None)
        transformer_encoder = self.pooling(transformer_encoder)

        #x = tf.concat([embedding_sex, embedding_age, embedding_current, transformer_encoder[:, -1, :]], 1)
        x = tf.concat([embedding_sex, embedding_age, embedding_current, transformer_encoder], 1)

        x = self.dense_2(x)
        x = tf.math.l2_normalize(x, axis=1)

        if training:
            z = self.output_layer(x)
            return z
        else:
            return x
