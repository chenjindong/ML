#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8

'''
rnn for mnist classification
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(acc,feed_dict={xs:v_xs}) # have problem
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return initial

def RNN(X, weights, biases):
    X = tf.reshape(X,[-1, n_inputs]) # (batch_size, time_steps, inputs_dim) => (batch_size*time_steps, inputs_dim)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) #(batch_size, time_step, input_dim)
    X_in = tf.transpose(X_in, [1,0,2]) # [time_step, batch_size, input_dim]
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32, time_major=True)
    outputs = tf.unstack(outputs) # remove first demension,[batch_size,hidden_size]
    
    results = tf.nn.softmax(tf.matmul(outputs[-1],weights['out']) + biases['out'])
    return results

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperarameters
lr = 0.001 
training_iters = 100000
batch_size = 128
n_inputs = 28   # dimensionality of input x 
n_steps = 28    # time step
n_hidden_units = 128
n_classes = 10

xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
ys = tf.placeholder(tf.float32, [None, n_classes])
weights = {'in':weight_variable([n_inputs, n_hidden_units]),
            'out':weight_variable([n_hidden_units,n_classes])}
biases = {'in':bias_variable([n_hidden_units]), 
        'out':bias_variable([n_classes])}

prediction = RNN(xs, weights, biases)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=ys))
#train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        x_batch,y_batch = mnist.train.next_batch(batch_size) # x_batch.shape(784)
        x_batch = x_batch.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_step, feed_dict={xs:x_batch, ys:y_batch})
        if i % 100 == 0:
            loss = sess.run(cross_entropy, feed_dict={xs:x_batch, ys:y_batch})
            acc = compute_accuracy(mnist.test.images.reshape([-1,n_steps,n_inputs]), mnist.test.labels)
            print('----epoch=%d, loss=%f, test accuracy=%f----' % (i,loss,acc))















