# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf 

x = tf.Variable(0)
s = tf.constant(2)
y = tf.add(x, s)

new_y = tf.assign(x, y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		result = sess.run(new_y)
		print(result)
