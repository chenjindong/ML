#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8

'''
rnn for mnist classification
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# hyperarameters
lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28   # dimensionality of input x
n_steps = 28    # time step
n_hidden_units = 100
n_classes = 10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(acc, feed_dict={xs: v_xs})  # have problem
    return result

def RNN(X):
    '''
    tip: input dimensions and hidden units of the first layer must be same
    :param X: (batch_size, time_steps, inputs_dim)
    :return prediction: (batch_size)
    '''
    #X = tf.reshape(X, [-1, n_inputs])
    #w1 = tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units], stddev=0.1),  dtype=tf.float32)
    #b1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_units]), dtype=tf.float32)
    #X_in = tf.matmul(X, w1) + b1
    #X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #X_in = tf.transpose(X_in, [1, 0, 2])  # swap dimension one and two
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # outputs (time_step=28, bach_size=128, hidden_unit=100)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, tf.transpose(X, [1,0,2]), dtype=tf.float32, time_major=True)
    outputs = tf.unstack(outputs)  # list of (batch_size,hidden_size)

    w2 = tf.Variable(tf.truncated_normal([n_hidden_units, n_classes], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.1, shape=[n_classes]), dtype=tf.float32)
    prediction = tf.nn.softmax(tf.matmul(outputs[-1], w2) + b2) # softmax是只用了最后一个time step的cell中的units
    return prediction

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])

prediction = RNN(xs)
cross_entropy = -tf.reduce_mean(ys*tf.log(prediction))
trainer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(batch_size) # x_batch.shape(784)
        x_batch = x_batch.reshape([batch_size, n_steps, n_inputs])
        sess.run(trainer, feed_dict={xs: x_batch, ys: y_batch})
        if i % 100 == 0:
            loss = sess.run(cross_entropy, feed_dict={xs: x_batch, ys: y_batch})
            acc = compute_accuracy(mnist.test.images.reshape([-1, n_steps, n_inputs]), mnist.test.labels)
            print('----iteration=%d, loss=%f, test accuracy=%f----' % (i, loss, acc))















