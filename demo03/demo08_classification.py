# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1. Number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
	Weight = tf.Variable(tf.random_normal([in_size, out_size])) 
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # biases not 0 is good
	Wx_plus_b = tf.matmul(inputs, Weight) + biases
	# if activation function is None or not:
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

# 2. Define palceholder for inputs to nerwork
xs = tf.placeholder(tf.float32, [None, 28*28]) # input num(every symbol is 784)
ys = tf.placeholder(tf.float32, [None, 10])    # output

# 3. Add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# 4. The error between prediction and real data
cross_entorpy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
				reduction_indices=[1]))  # loss(“交叉熵”)
train_setp = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entorpy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# compute
def compute_accuracy(v_xs, v_ys):
	global prediction # Needed when changed it's value. 
	y_pre = sess.run(prediction, feed_dict={xs: v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) # pre vs. ys
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
	return result


for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100) # 每次只取100张图片
	sess.run(train_setp, feed_dict={xs: batch_xs, ys: batch_ys})
	if 0 == i % 50:
		# There has test data & training data
		print(compute_accuracy(mnist.test.images, mnist.test.labels))
