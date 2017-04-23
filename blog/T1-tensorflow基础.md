## TensorFlow 基础

刚了解 Google 的 tensorflow, 使用的是 Python 版本的. 简单的整理下基本用法.

### 1.  常量(constant)

首先呢, 需要导入 tensorflow:
```
import tensorflow as tf
```
```
# 在 Python 中定义常量如下:
x = 10
# 在 tf 中则是这样的:
x = tf.constant(10)
```

### 2. 变量(variable)

定义变量, 赋初值(<font color="#FF8C00">必须初始化</font>), 并给出变量的名字:
```
# 直接定义
y1 = tf.Variable(0)
# 或者加上变量名
y2 = tf.Variable(0, name='counter')
```
当然, 在图像处理中, 常用的矩阵定义也是一样的:
```
# 定义 3x3 全1矩阵
m1 = tf.Variable(tf.ones([3, 3]))
# 定义 3x3 全0矩阵, 并给出变量名
m2 = tf.Variable(tf.zeros([3, 3]), name='matrix')
```
变量定义后, 必须显式的初始化 :  `init = tf.global_variables_initializer()`.

### 3. 占位符(placeholder)
在上面定义的「变量」必须初始化, 但是有些变量我们刚开始并不知道, 要进过计算之后才能得到. 这就需要「占位符」来 hold 住, 先替我们占个位置.
>函数原型如下:
tf.placeholder(dtype, shape=None, name=None)

```
x = tf.palceholder(tf.float32, [None, 512])
```
这里指定了占位符 x 变量类型为 float32, 形状 shape 为任意维度, 大小为 512 的张量, 如果 shape 没有指定的话可以为空 「None」. 这里没有定义变量名.
* 后面需要用到占位符时, 要注意必须向其填充值「feed_dict」.

### 4. 会话(Session)
tf不会去一条条地执行各个操作, 而是把所有的操作都放入到一个图「graph」中, 图中的每一个结点就是一个操作. 然后行将整个graph 的计算过程交给TensorFlow 的Session, Session 可以运行整个计算过程. (与之相似的还有 InteractiveSession )
```
# 创建会话
sess = tf.Session()
# 关闭会话 (当然 用 with...as 更方便)
sess.close()
```

***

### 示例
</br>

示例1: 使用常量实现 tf 加法 (8 + 2)
```
x1 = tf.constant(8, name='x_value')
x2 = tf.constant(2)

y = tf.add(x1, x2)

with tf.Session() as sess:
	result = sess.run(y)
	print(result)
```
输出结果为: `10` 
</br>

示例2: 利用常量和变量实现 tf 加法:
```
import tensorflow as tf 

x = tf.Variable(0)
s = tf.constant(2)
y = tf.add(x, s)

# 这里使用的 assign 是用来更新变量 x 的, 我们共循环 3 次
# 每次把 y 的值更新到变量 x 中, 实现 0+2 | 2+2 | 2+2+2
new_y = tf.assign(x, y)

# 这里非常重要, 必须要初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	# 以上的定义都在 Session 里真正计算
	sess.run(init)
	for _ in range(3):
		result = sess.run(new_y)
		print(result)
```
输出结果为: `2 4 6`
