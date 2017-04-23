# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # Display
import tkinter

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


# Create data: [-1, 1] steps: 300 newaxis: weidu--have 300 examples.
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# shape: equal to x_data type. 0.05: variance(fang cha)
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入
# None: input number unlimited, 1: one feature
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 输出层和输入层的结构是一样的；隐藏层我们可以自己假设
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # Hidden layer
prediction = add_layer(layer1, 10, 1, activation_function=None) # Output layer

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
	reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Plot.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()   # Don't stop (old python: plt.show(block=False))
plt.show()  # Show & stop running

# 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入
# Start train
for i in range(1000):
	# training
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
	if 0 == i % 50:
		# print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction, feed_dict={xs: x_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # x p color width
		plt.pause(0.2)
sess.close()
		