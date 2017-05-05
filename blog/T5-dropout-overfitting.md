## Dropout 解决 overfitting
</br>

相对于过拟合（overfitting，或称：过度学习）是指，使用过多参数，以致太适应训练数据而非一般情况；另一种常见的现象是使用太少参数，以致于不适应当前的训练数据，这则称为欠拟合（underfitting，或称：拟合不足）现象。<sup>[2]</sup>
</br>

防止过拟合，我们需要用到一些方法，如：early stopping、数据集扩增（Data augmentation）、正则化（Regularization）、Dropout等。<sup>[3]</sup>
</br>

本次数据来自 `sklearn`, 首先导入模块
```
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
```
</br>

在之前代码的基础上修改, 增加 `keep_prob` 占位符保留数据的概率
```
# k = 1, 保留 100%, 即没有 dropout 任何数据.
keep_prob = tf.placeholder(tf.float32)
```
</br>

准备训练数据（train）测试数据（test）
```
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
```
</br>

在训练过程中，overfitting 的问题与 keep_prob 相关，keep_prob = 1 没有dropout 任何数据， keep_prob = 0.5 则能明显看出 dropout 的效果。
</br>
** keep_prob = 1 **
![](https://github.com/TaylorBoy/tensorflow-my/blob/master/blog/images/tensorboard-8.png?raw=true)
</br>

** keep_prob = 0.5 **
![](https://github.com/TaylorBoy/tensorflow-my/blob/master/blog/images/tensorboard-7.png?raw=true)
</br>

***

### 完整代码
</br>
```
# !/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data     # img data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)  # +++
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)		 # +++
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)     # +++
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)  # +++
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})  # +++
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})  
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)   # +++

```

</br>

### Reference
[1] 莫烦Python: [Dropout 解决 overfitting](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-02-dropout/)
[2] 拾毅者: [机器学习—过拟合overfitting](http://blog.csdn.net/dream_angel_z/article/details/48898817)
[3] 一只鸟的天空: [机器学习中防止过拟合的处理方法](http://blog.csdn.net/heyongluoyao8/article/details/49429629)