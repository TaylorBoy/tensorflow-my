# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np  

# 1. Create data.
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### 2. Create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # demension, start, end
# Start with biases=0
biases = tf.Variable(tf.zeros([1]))  

# Prediction.
y = Weights * x_data + biases  


# Loss
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate: 0.5 < 1
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

### 2. Create tensorflow structure end ###

sess = tf.Session()  # Create Session.
sess.run(init)   # Important.

for step in range(200):
    sess.run(train)
    if 0 == step % 20:
        print(step, sess.run(Weights), sess.run(biases))
