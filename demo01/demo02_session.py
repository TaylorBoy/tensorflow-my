# !/usr/bin/env python3
# -*-coding: utf-8 -*-

import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], 
		       [2]])
product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

# Constant, add function:
x1 = tf.constant(8, name='x_value')
x2 = tf.constant(2)

y = tf.add(x1, x2)

with tf.Session() as sess:
	result = sess.run(y)
	print(result)

