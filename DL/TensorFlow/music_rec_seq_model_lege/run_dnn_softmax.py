# -*- coding: utf-8 -*-
# @File    : run_dnn.py
# @Author  : letian
# @Mail    : @.com
# @Time    : 2019/11/15 10:38

import tensorflow as tf
import os
from absl import app
from absl import flags

import src_new.model.seq_cnn_softmax as seq_cnn_softmax
import src_new.model.seq_dnn_softmax as seq_dnn_softmax
import src_new.model.seq_rnn_softmax as seq_rnn_softmax
import src_new.model.seq_rnn_softmax_no_current as seq_rnn_softmax_no_current
import src_new.model.seq_rnn_softmax_no_profile as seq_rnn_softmax_no_profile
import src_new.model.seq_rnn_softmax_no_rate as seq_rnn_softmax_no_rate
import src_new.model.seq_rnn_softmax_no_time_category as seq_rnn_softmax_no_time_category
import src_new.model.seq_transformer_softmax as seq_transformer_softmax
import src_new.model.seq_transformer_softmax_no_fix as seq_transformer_softmax_no_fix
import src_new.model.seq_cnn_softmax_deep as seq_cnn_softmax_deep
import src_new.utils.pretrain_embedding as pretrain_embedding

FLAGS = flags.FLAGS

flags.DEFINE_string("model_type", "rnn", "model type.")
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('neg_rate', 0.8, 'Negative rate')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to run trainer.')
flags.DEFINE_integer('song_size', 3905812, 'song vocab size')
flags.DEFINE_integer('singer_size', 906700, 'singer vocab size')
flags.DEFINE_integer('sex_size', 5, 'sex vocab size')
flags.DEFINE_integer('age_size', 20, 'age vocab size')
flags.DEFINE_integer('time_category_size', 8, 'time category vocab size')
flags.DEFINE_integer('rate_category_size', 11, 'rate category vocab size')
flags.DEFINE_integer('default_dim', 50, 'default dimension.')
flags.DEFINE_integer('dense_dim', 50, 'Dense dimension.')
flags.DEFINE_integer('seq_length', 20, 'Sequence length.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('shuffle_buffer_size', 20000, 'Shuffle batch size')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_string('model_dir', 'models', 'Directory to save the model.')
flags.DEFINE_string('pretrain_embedding', 'pretrain_embedding', 'Directory to load pretrain embedding')


def MainSeqModel(argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_path = os.path.join(FLAGS.train_dir, "train")
    dev_path = os.path.join(FLAGS.train_dir, "dev")
    test_path = os.path.join(FLAGS.train_dir, "test")
    model_path = os.path.join(FLAGS.model_dir, "model")
    checkpoint_path = os.path.join(FLAGS.model_dir, "checkpoint/cp-{epoch:04d}.ckpt")

    train_data, dev_data, test_data = create_record_dataset(train_path, dev_path, test_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     save_freq='epoch')

    print("load pretrain embedding...")
    pre_embedding = pretrain_embedding.load_embedding(FLAGS.pretrain_embedding, FLAGS.song_size, FLAGS.default_dim)
    print("finish load pretrain embedding")

    model = ModelFactory(FLAGS.model_type, pre_embedding)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    # loss_updater = tf.keras.losses.BinaryCrossentropy()
    # train_accuracy = tf.keras.metrics.BinaryAccuracy()

    loss_updater = tf.keras.losses.SparseCategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for i in range(FLAGS.num_epochs):
        for features, labels in train_data:
            loss, accuracy = train_one_step(model, optimizer, loss_updater, train_accuracy, features, labels)
            print('epoch', i, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

    tf.saved_model.save(model, model_path)
    model.save_weights(model_path + ".h5")


def ModelFactory(model_type, pre_embedding):
    if model_type == "cnn_softmax":
        return seq_cnn_softmax.SeqModelCNNSoftmax(default_dim=FLAGS.default_dim,
                                                  seq_length=FLAGS.seq_length,
                                                  song_size=FLAGS.song_size,
                                                  singer_size=FLAGS.singer_size,
                                                  sex_size=FLAGS.sex_size,
                                                  age_size=FLAGS.age_size,
                                                  time_category_size=FLAGS.time_category_size,
                                                  rate_category_size=FLAGS.rate_category_size,
                                                  pre_embedding=pre_embedding)
    elif model_type == "rnn_softmax":
        return seq_rnn_softmax.SeqModelRNNSoftmax(default_dim=FLAGS.default_dim,
                                                  seq_length=FLAGS.seq_length,
                                                  song_size=FLAGS.song_size,
                                                  singer_size=FLAGS.singer_size,
                                                  sex_size=FLAGS.sex_size,
                                                  age_size=FLAGS.age_size,
                                                  time_category_size=FLAGS.time_category_size,
                                                  rate_category_size=FLAGS.rate_category_size,
                                                  pre_embedding=pre_embedding)
    elif model_type == "dnn_softmax":
        return seq_dnn_softmax.SeqModelDNNSoftmax(default_dim=FLAGS.default_dim,
                                                  seq_length=FLAGS.seq_length,
                                                  song_size=FLAGS.song_size,
                                                  singer_size=FLAGS.singer_size,
                                                  sex_size=FLAGS.sex_size,
                                                  age_size=FLAGS.age_size,
                                                  time_category_size=FLAGS.time_category_size,
                                                  rate_category_size=FLAGS.rate_category_size,
                                                  pre_embedding=pre_embedding)
    elif model_type == "cnn_softmax_deep":
        return seq_cnn_softmax_deep.SeqModelCNNSoftmaxDeep(default_dim=FLAGS.default_dim,
                                                           seq_length=FLAGS.seq_length,
                                                           song_size=FLAGS.song_size,
                                                           singer_size=FLAGS.singer_size,
                                                           sex_size=FLAGS.sex_size,
                                                           age_size=FLAGS.age_size,
                                                           time_category_size=FLAGS.time_category_size,
                                                           rate_category_size=FLAGS.rate_category_size,
                                                           pre_embedding=pre_embedding)
    elif model_type == "rnn_softmax_no_current":
        return seq_rnn_softmax_no_current.SeqModelRNNSoftmaxNoCurrent(default_dim=FLAGS.default_dim,
                                                                      seq_length=FLAGS.seq_length,
                                                                      song_size=FLAGS.song_size,
                                                                      singer_size=FLAGS.singer_size,
                                                                      sex_size=FLAGS.sex_size,
                                                                      age_size=FLAGS.age_size,
                                                                      time_category_size=FLAGS.time_category_size,
                                                                      rate_category_size=FLAGS.rate_category_size,
                                                                      pre_embedding=pre_embedding)
    elif model_type == "rnn_softmax_no_profile":
        return seq_rnn_softmax_no_profile.SeqModelRNNSoftmaxNoProfile(default_dim=FLAGS.default_dim,
                                                                      seq_length=FLAGS.seq_length,
                                                                      song_size=FLAGS.song_size,
                                                                      singer_size=FLAGS.singer_size,
                                                                      sex_size=FLAGS.sex_size,
                                                                      age_size=FLAGS.age_size,
                                                                      time_category_size=FLAGS.time_category_size,
                                                                      rate_category_size=FLAGS.rate_category_size,
                                                                      pre_embedding=pre_embedding)
    elif model_type == "rnn_softmax_no_rate":
        return seq_rnn_softmax_no_rate.SeqModelRNNSoftmaxNoRate(default_dim=FLAGS.default_dim,
                                                                seq_length=FLAGS.seq_length,
                                                                song_size=FLAGS.song_size,
                                                                singer_size=FLAGS.singer_size,
                                                                sex_size=FLAGS.sex_size,
                                                                age_size=FLAGS.age_size,
                                                                time_category_size=FLAGS.time_category_size,
                                                                rate_category_size=FLAGS.rate_category_size,
                                                                pre_embedding=pre_embedding)
    elif model_type == "rnn_softmax_no_time_category":
        return seq_rnn_softmax_no_time_category.SeqModelRNNSoftmaxNoTimeCategory(default_dim=FLAGS.default_dim,
                                                                                 seq_length=FLAGS.seq_length,
                                                                                 song_size=FLAGS.song_size,
                                                                                 singer_size=FLAGS.singer_size,
                                                                                 sex_size=FLAGS.sex_size,
                                                                                 age_size=FLAGS.age_size,
                                                                                 time_category_size=FLAGS.time_category_size,
                                                                                 rate_category_size=FLAGS.rate_category_size,
                                                                                 pre_embedding=pre_embedding)
    elif model_type == "transformer_softmax":
        return seq_transformer_softmax.SeqModelTransformerSoftmax(default_dim=FLAGS.default_dim,
                                                                  seq_length=FLAGS.seq_length,
                                                                  song_size=FLAGS.song_size,
                                                                  singer_size=FLAGS.singer_size,
                                                                  sex_size=FLAGS.sex_size,
                                                                  age_size=FLAGS.age_size,
                                                                  time_category_size=FLAGS.time_category_size,
                                                                  rate_category_size=FLAGS.rate_category_size,
                                                                  pre_embedding=pre_embedding)
    elif model_type == "transformer_softmax_no_fix":
        return seq_transformer_softmax_no_fix.SeqModelTransformerSoftmaxNoFix(default_dim=FLAGS.default_dim,
                                                                              seq_length=FLAGS.seq_length,
                                                                              song_size=FLAGS.song_size,
                                                                              singer_size=FLAGS.singer_size,
                                                                              sex_size=FLAGS.sex_size,
                                                                              age_size=FLAGS.age_size,
                                                                              time_category_size=FLAGS.time_category_size,
                                                                              rate_category_size=FLAGS.rate_category_size,
                                                                              pre_embedding=pre_embedding)
    else:
        return None


@tf.function
def train_one_step(model, optimizer, loss_updater, train_accuracy, features, labels):
    with tf.GradientTape() as tape:
        # labels_pred, _ = model(features, True)
        labels_pred = model(features, True)
        loss = loss_updater(y_true=labels, y_pred=labels_pred)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

    train_accuracy.update_state(y_true=labels, y_pred=labels_pred)

    return loss, train_accuracy.result()


feature_description = {
    'binary_label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'predict_label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'pos_label': tf.io.FixedLenFeature([5], tf.int64),
    'neg_label': tf.io.FixedLenFeature([20], tf.int64),
    'sex': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'age': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'current_song': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'current_time_category': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'current_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'current_singer': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'song': tf.io.FixedLenFeature([20], tf.int64),
    'time_category': tf.io.FixedLenFeature([20], tf.int64),
    'rate': tf.io.FixedLenFeature([20], tf.int64),
    'singer': tf.io.FixedLenFeature([20], tf.int64)
}


def random_get_predict_label(features):
    predict_labels = features['pos_label']

    random_val = tf.keras.backend.random_uniform([1], 0, 1)
    id = int(predict_labels.get_shape()[0] * random_val[0])

    predict_label = tf.cond(tf.math.equal(predict_labels[id], tf.constant([0], dtype=tf.int64, shape=[1])),
                            lambda: predict_labels[0], lambda: predict_labels[id])

    return predict_label


@tf.function
def parse_record(record_proto):
    features = tf.io.parse_single_example(record_proto, feature_description)

    #predict_label = random_get_predict_label(features)
    predict_label = features['pos_label'][0]

    features.pop("pos_label")
    features.pop("neg_label")

    return features, predict_label


def list_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def create_record_dataset(train_path, dev_path, test_path):
    train_files = list_files(train_path)
    dev_files = list_files(dev_path)
    test_files = list_files(test_path)

    train_data = tf.data.TFRecordDataset(train_files)
    dev_data = tf.data.TFRecordDataset(dev_files)
    test_data = tf.data.TFRecordDataset(test_files)

    train_data = train_data.shuffle(buffer_size=FLAGS.shuffle_buffer_size)

    train_data = train_data.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dev_data = dev_data.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_data = train_data.batch(FLAGS.batch_size)
    dev_data = dev_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dev_data = dev_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_data, dev_data, test_data


if __name__ == "__main__":
    app.run(MainSeqModel)
