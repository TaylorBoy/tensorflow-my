# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

# 定义一个变量，给出初值，可以加变量名
state = tf.Variable(0, name='counter')

# 定义一个常量
one = tf.constant(1)

# 计算：加法（并没有直接计算）
new_value = tf.add(state, one)

# 将 state 更新成 new_value
update = tf.assign(state, new_value)

# 如果定义了 Variable，就一定要 initialize
init = tf.global_variables_initializer()

# 使用 Session
with tf.Session() as sess:
    sess.run(init)    # 这里真正激活
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
 
