import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = "output"

max_epoch = 100

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256


def build_generator(z_prior):
    '''
    z_prior: (batch_size, 100)
    x_generate: (batch_size, 784)
    function G: z => image
    '''
    with tf.variable_scope('generator'):
        h1 = tf.layers.dense(z_prior, h1_size, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, h2_size, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, img_size, activation=tf.nn.tanh)
        return h3

def build_discriminator(x_data, x_generated, keep_prob):
    '''
    x_data, x_generated: (batch_size, 784)
    y_data, y_generated: (batch_size, 1)
    function D: x_data => y_data
                x_generated => y_generated
    '''
    with tf.variable_scope('discriminator'):
        x_in = tf.concat([x_data, x_generated], 0)

        h1 = tf.layers.dense(x_in, h2_size, tf.nn.relu)
        h1 = tf.nn.dropout(h1, keep_prob)

        h2 = tf.layers.dense(h1, h1_size, tf.nn.relu)
        h2 = tf.nn.dropout(h2, keep_prob)

        h3 = tf.layers.dense(h2, 1, tf.nn.sigmoid)  # (batch_size*2, 1)
        h3 = tf.reshape(h3, [-1])  # (batch_size*2, )

        y_data = tf.slice(h3, [0], [batch_size])  # (batch_size, )
        y_generated = tf.slice(h3, [batch_size], [-1])  # # (batch_size, )
    return y_data, y_generated


def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # placeholder
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # G and D
    x_generated = build_generator(z_prior)  # x_generated:(batch_size, 784)
    y_data, y_generated = build_discriminator(x_data, x_generated, keep_prob)  # y:(batch_size, 1)

    d_loss = - tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.reduce_mean(tf.log(y_generated))

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    g_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # summary
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)
    summary_all = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs', sess.graph)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)  # remove file recursively
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    for i in range(sess.run(global_step), max_epoch):
        print("-----------------epoch:%s--------------------" % (i))
        for j in range(int(60000 / batch_size)):
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            sess.run([d_trainer], feed_dict={x_data: x_value, z_prior: z_value, keep_prob: 0.7})  # train D
            if j % 1 == 0:  # %1是什么鬼
                sess.run([g_trainer], feed_dict={x_data: x_value, z_prior: z_value, keep_prob: 0.7})  # train G

            train_summary = sess.run(summary_all, feed_dict={x_data: x_value, z_prior: z_value, keep_prob: 0.7})
            train_writer.add_summary(train_summary, i*int(60000 / batch_size)+j)

        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, os.path.join(output_path, "sample%s.jpg" % i))
        # z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        # x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        # show_result(x_gen_val, os.path.join(output_path, "random_sample%s.jpg" % i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, os.path.join(output_path, "test_result.jpg"))


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()