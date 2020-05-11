import numpy as np
import tensorflow as tf
import cjdpy
import random
import time
import os
import sys
from make_data import get_train_and_eval_data, get_predict_data, load_vocab

FIRST_CATEGORY_DIM = 5
SECOND_CATEGORY_DIM = 10
TAG_DIM = 50
MEDIA_DIM = 50
RATE_DISCRETIZE_DIM = 5
POSITION_DIM = 8

FIRST_CATEGORY_SIZE = 37
SECOND_CATEGORY_SIZE = 224
TAG_SIZE = 72193 #  67764
MEDIA_SIZE = 94386 #  87158
RATE_DISCRETIZE_SIZE = 5
POSITION_SIZE = 26

SEQUENCE_LENGTH = 26
TAG_SEQUENCE_LENGTH = 6

data_path = "../data/dataset.txt.middle"
model_path = "output"
save_checkpoints_steps = 5000
batch_size = 64
epoch = 5
sequence_encoding = "average"  # GRU, self-attention, attention, average
user_video_combine_method = "cosine" #  concat, cosine

def model_fn(features, labels, mode):
    first_category_lookup = tf.Variable(tf.truncated_normal(shape=[FIRST_CATEGORY_SIZE, FIRST_CATEGORY_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)
    second_category_lookup = tf.Variable(tf.truncated_normal(shape=[SECOND_CATEGORY_SIZE, SECOND_CATEGORY_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)
    tag_lookup = tf.Variable(tf.truncated_normal(shape=[TAG_SIZE, TAG_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)
    media_lookup = tf.Variable(tf.truncated_normal(shape=[MEDIA_SIZE, MEDIA_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)
    rate_discretize_lookup = tf.Variable(tf.truncated_normal(shape=[RATE_DISCRETIZE_SIZE, RATE_DISCRETIZE_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)
    position_lookup = tf.Variable(tf.truncated_normal(shape=[POSITION_SIZE, POSITION_DIM], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None), trainable=True)

    first_category = tf.nn.embedding_lookup(first_category_lookup, features["first_category"])
    second_category = tf.nn.embedding_lookup(second_category_lookup, features["second_category"])
    media = tf.nn.embedding_lookup(media_lookup, features["media"])
    tag = tf.nn.embedding_lookup(tag_lookup, features["tag"])
    rate_discretize = tf.nn.embedding_lookup(rate_discretize_lookup, features["rate_discretize"])
    position = tf.nn.embedding_lookup(position_lookup, features["position"])

    print(type(features))
    print("first_category: ", first_category)
    print("second_category", second_category)
    print("media: ", media)
    print("tag: ", tag)
    print("rate_discretize: ", rate_discretize)
    print("position", position)

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("weight", features["weight"])
        print("labels", labels)

    tag = tf.reduce_mean(tag, 2)
    print("tf.reduce_mean(tag, 2)", tag)

    # user embedding
    user_embedding = tf.concat([first_category[:, :-1, :], second_category[:, :-1, :], media[:, :-1, :], tag[:, :-1, :],
                                rate_discretize[:, :-1, :]], 2)
    print("user behavior seq shape: ", user_embedding)

    # video embedding
    vid_embedding = tf.concat([first_category[:, -1, :], second_category[:, -1, :], media[:, -1, :], tag[:, -1, :]], 1)
    vid_embedding = tf.layers.dense(vid_embedding, 100, "selu")  # (32, 80, )
    print("origin vid embedding: ", vid_embedding)
    
    if sequence_encoding == "average":
        user_embedding = tf.concat([user_embedding, position[:, :-1, :]], 2)
        user_embedding = tf.layers.dense(user_embedding, 100, "selu")  # new add
        print("user embedding after dense layer: ", user_embedding)  # (_, 25, 100)
        user_embedding = tf.reduce_mean(user_embedding, 1)
        print("tf.reduce_mean(user_embedding, 1)", user_embedding)
    elif sequence_encoding == "GRU":
        # user embedding by GRU
        user_embedding = tf.layers.dense(user_embedding, 120, "selu")  # new add
        user_embedding = tf.keras.layers.GRU(120)(user_embedding)
        print("user behavior sequence after GRU: ", user_embedding)
        user_embedding = tf.layers.dense(user_embedding, 100, "selu")  # (32, 100, )
    elif sequence_encoding == "self-attention":
        # user embdding by self-attention
        user_embedding = tf.concat([user_embedding, position[:, :-1, :]], 2)
        user_embedding = tf.layers.dense(user_embedding, 100, "selu")  # new add
        print("user embedding after dense layer: ", user_embedding)  # (_, 25, 100)

        QK = tf.matmul(user_embedding, tf.transpose(user_embedding, [0, 2, 1]))
        QK = QK/np.sqrt(100)
        print("QK", QK)
        QK = tf.keras.layers.Softmax(axis=0)(QK)
        user_embedding = tf.matmul(QK, user_embedding)
        print("user embedding after self-attention", user_embedding)
        user_embedding = tf.reduce_mean(user_embedding, axis = 1) # should be (32,120), need to verify
        print("max pooling", user_embedding)
        user_embedding = tf.layers.dense(user_embedding, 100, "selu")  # (32, 80, )
    elif sequence_encoding == "attention":
        # attention between targe video and videos in user behavior sequence
        user_embedding = tf.concat([user_embedding, position[:, :-1, :]], 2)
        user_embedding = tf.layers.dense(user_embedding, 100, "selu")
        print("user embedding after dense layer: ", user_embedding)  # (_, 25, 100)

        query = tf.expand_dims(vid_embedding, 1)  # (_, 1, 100)
        print("tf.expand_dims(vid_embedding, 0)", query)
        query = tf.tile(input=query, multiples=[1,25,1])  # (_, 25, 100)
        print("tf.tile(input=query, multiples=[1,25,1])", query)

        query = tf.concat([query, user_embedding], -1)  # (_, 25, 200)
        print("tf.concat([query, user_embedding], 1)", query)

        query = tf.layers.dense(query, 150, "tanh")  # (_, 25, 150)
        print("tf.layers.dense(query, 150, tanh)", query)

        query = tf.layers.dense(query, 1)  # (_, 25, 1)
        print("tf.layers.dense(query, 1)", query)
        query = tf.squeeze(query)  # (_, 25)
        print("tf.squeeze(query)", query)

        query = tf.keras.layers.Softmax()(query)  # (_, 25)

        query = tf.expand_dims(query, -1)  # (_, 25, 1)
        query = tf.tile(query, [1, 1, 100])  # (_, 25, 100)
        print("expand_dims & tile", query)

        user_embedding = tf.multiply(query, user_embedding)  # (_, 25, 100)
        user_embedding = tf.reduce_mean(user_embedding, 1)  # (_, 100)
        print("tf.reduce_mean(user_embedding, 1)", user_embedding)

    if user_video_combine_method == "concat":
        # concat user and vid embedding
        user_vid_mix = tf.concat([user_embedding, vid_embedding], -1)
        user_vid_mix = tf.layers.dense(user_vid_mix, 50, "selu")
        logits = tf.layers.dense(user_vid_mix, 2)
    elif user_video_combine_method == "cosine":
        # UWV
        user_embedding = tf.layers.dense(user_embedding, 100)  # (_, 100)
        user_vid_mix = tf.multiply(user_embedding, vid_embedding)  # (_, 100)
        logits = tf.reduce_sum(user_vid_mix, -1)
        print("logits, ", logits)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if user_video_combine_method == "cosine":
            labels = tf.cast(labels, tf.float32)
            loss_ori = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(tf.multiply(features["weight"], loss_ori))
        else:
            loss_ori = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # loss = tf.reduce_mean(loss_ori)
            loss = tf.reduce_mean(tf.multiply(features["weight"], loss_ori))
        
        global_step = tf.train.get_global_step() # 获取训练全局参数step
        optimizer = tf.train.AdamOptimizer(0.001)
        train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1)) # 将优化器和全局step的累加方法打包成一个方法组，相当于把若干个方法打包成事务执行的模式

        if user_video_combine_method == "cosine":
            predictions = tf.where(logits >= 0.5, tf.ones_like(logits), logits)
            predictions = tf.where(predictions <= 0.5, tf.zeros_like(predictions), predictions)
        else:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        precision = tf.metrics.precision(labels=labels, predictions=predictions)
        recall = tf.metrics.recall(labels=labels, predictions=predictions)
        auc = tf.metrics.auc(labels=labels, predictions=predictions)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss, 
            eval_metric_ops={"eval_acc": accuracy, "eval_precision": precision, "eval_recall": recall, "eval_auc": auc},
            train_op=train)

    if mode == tf.estimator.ModeKeys.PREDICT:
        if user_video_combine_method == "cosine":
            predictions = {  # for export model
                "y": tf.nn.sigmoid(logits),
                "ue": user_embedding,
                "ie": vid_embedding
            }
        else:
            predictions = {  # for export model
                #"y": tf.argmax(logits, axis=1)
                "y": tf.nn.softmax(logits),
            }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

def serving_input_fn():
    # save to pb
    first_category = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='first_category')
    second_category = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='second_category')
    media = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='media')
    tag = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH, TAG_SEQUENCE_LENGTH], name='tag')
    rate_discretize = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='rate_discretize')
    position = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='position') 

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'first_category': first_category,
        'second_category': second_category,
        'media': media,
        'tag': tag,
        'rate_discretize': rate_discretize,
        'position': position,
    })()
    return input_fn


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("input like: python test.py train/predict/export")
        assert False

    mode = sys.argv[1]

    tf.logging.set_verbosity(tf.logging.INFO)

    # 设置save_checkpoints_steps才会进行evaluation（只有save checkpoint的时候才会evaluation）
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps, keep_checkpoint_max=5, log_step_count_steps=10000)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_path, config=run_config)

    if mode == "train":
        train_input_x, eval_input_x, y_train, y_eval = get_train_and_eval_data(data_path)

        print(y_train[:100])
        print(train_input_x["weight"][:100])
        print(train_input_x["position"][:1])
        print(train_input_x["first_category"][0])
        print(train_input_x["second_category"][0])
        print(train_input_x["tag"][0])
        print(train_input_x["media"][0])
        print(train_input_x["rate_discretize"][0])

        train_input_fn = tf.estimator.inputs.numpy_input_fn(train_input_x, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(eval_input_x, y_eval, batch_size=batch_size, shuffle=False)
        
        MAXSTEPS = len(y_train)*epoch//batch_size

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=MAXSTEPS)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, start_delay_secs=0, throttle_secs=0)  # modify

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif mode == "predict":
        x, y = get_predict_data("../data/pred.txt")

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=batch_size, shuffle=False)
        
        _, id2tag = load_vocab("../vocab/tag_vocab.txt")
        _, id2fc = load_vocab("../vocab/first_category_vocab.txt")
        _, id2sc = load_vocab("../vocab/second_category_vocab.txt")
        _, id2media = load_vocab("../vocab/media_vocab.txt")

        res = list(estimator.predict(input_fn=pred_input_fn))

        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        for i in range(20):
            print("prob: ", res[i]["y"])
            for j, tag_list in enumerate(x["tag"][i]):
                tags = [id2tag[tag] for tag in tag_list]
                print(id2fc[x["first_category"][i][j]], id2sc[x["second_category"][i][j]], tags, id2media[x["media"][i][j]], x["rate_discretize"][i][j])
            logits = np.sum(res[i]["ue"]*res[i]["ie"])
            print(logits, sigmoid(logits))
            print("\n")

    elif mode == "export":
        estimator.export_savedmodel("output/pb_model", serving_input_fn)


