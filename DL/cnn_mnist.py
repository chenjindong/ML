#!/home/gdmlab/anaconda3/bin/python
#coding:utf-8

'''
cnn for mnist classification
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs, keep_prob:0.5})
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W,strides=[1,1,1,1], padding='SAME') # stride [1,x_movement,y_movement,1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # x_image.shape [n_sample,28,28,1]

# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32]) # patch(5*5), in_size(channel), out_size(feature map)
b_conv1 = bias_variable([32]) # one filter corresponding one bias, one feature map
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # 14*14*32
# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64]) # patch(5*5), in_size(feature map), out_size(feature map)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))+b_conv2 # 14*14*64
h_pool2 = max_pool_2x2(h_conv2) # 7*7*64
# full connect 1
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64]) # flatten
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# full connect 2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        x_batch,y_batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:x_batch, ys:y_batch, keep_prob:0.8})
        if i % 100 == 0:
            loss = sess.run(cross_entropy, feed_dict={xs:x_batch,ys:y_batch, keep_prob:0.5})
            acc = compute_accuracy(mnist.test.images,mnist.test.labels)
            print('----step=%d, loss=%f, accuracy=%f----' % (i,loss,acc))















