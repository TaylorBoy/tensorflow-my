## Tensorboard 可视化之图层
</br>

Tensorflow 自带的 tensorboard 可以构建我们的神经网络图层, 让我们看看自己的神经网络长什么样. 
</br>
###开始构建图层结构啦
</br>
我们要用到前面用到的代码来构建神经网络图像
</br>

首先是数据的输入 `input` :
```
# 我们先给输入和输出的占位符指定名称
# 指定的名称会在可视化的图层 input 中显示
xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
ys = tf.placeholder(tf.float32, [None, 1], name='y_in')
```
</br>

图层可以包含子图层, 所以, 我们要用到 `with tf.name_scope('inputs')` 将`xs`和`ys`包含起来, 作为输入层. (inputs 就是图层的名字, 可任意命名)
```
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_in')
```
</br>

接下来, 就是`layer`了, 我们前面用了`add_layer`函数来添加图层, 这里我可以直接在`add_layer`函数里面构建图层结构. ( 记得 name_scope 可以嵌套的哦
```
def add_layer(inputs, in_size, out_size, activation_function=None):
	# 每一个图层名为 `layer`
	with tf.name_scope('layer'):
		# 添加层里面的小部件也需要定义
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]))
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		with tf.name_scope('wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b, )
		return outputs	
```
</br>

最后是`loss`和`training`部分了, 同样为他们各自取名
```
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
							reduction_indices=[1]))
with tf.name_scope('train'):
	    train_setp = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
```
</br>

### 绘制我们的神经网络图啦
</br>

绘制方法为`tf.summary.FileWriter(dir, sess.graph)`, 第一个参数为图的储存路径, 第二个参数是将前面定义好的框架信息收集起来, 最后放到`dir`目录中, 因此需要先获得 Session
```
sess = tf.Session()
# 执行 python3 filename.py 之后会自动创建 graph 文件夹
# 并把生成的图层图像信息保存在 graph 下, 需要用浏览器观看
writer = tf.summary.FileWriter('graph/', sess.graph)
```
</br>

运行完整代码后「[完整代码] [url]」, 会自动生成图片信息并保存到 `graph` 目录中, 然后什么在 `graph` 上一级目录执行下面这条命令, 它会输出一条地址, 我们在浏览器上打开`http://127.0.1.6006:1`
```
Ubuntu ~#  tensorboard --logdir='./graph/'
Starting TensorBoard b'41' on port 6006
(You can navigate to http://127.0.1.1:6006)
...
```
这个网页有多个选项卡, 因为我们只定义了`sess.graph`, 所以我们切换到`GRAPH`, 可以看到我们的神经网络的基本结构


***

### 完整代码
[url]: http://taylor.easy.tensorflow.graph "code"
 
