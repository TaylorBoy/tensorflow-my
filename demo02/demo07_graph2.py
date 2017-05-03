# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


# Define add layer function.
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # Draw histogram: name, variable
            tf.summary.histogram(layer_name + '/weights', Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )

        # at histogram
        tf.summary.histogram(layer_name + '/output', outputs)
        return outputs

# Define palceholder for inputs to network.
# Use [with] including xs & ys:
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in') # Add name
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')

# Make up some real data
x_data = np.linspace(-1, -1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


# Add hidden layer
with tf.name_scope('hidden_layer'):
    l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# Add output layer
with tf.name_scope('output_layer'):
    prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# The error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
        reduction_indices=[1]))
    # Scalar -- at event
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_setp = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

merged = tf.summary.merge_all()

# ** Add frame to file
writer = tf.summary.FileWriter('./graph/', sess.graph)

# Important step
sess.run(tf.global_variables_initializer())

# Start training:
for i in range(1000):
    sess.run(train_setp, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(result, i)
