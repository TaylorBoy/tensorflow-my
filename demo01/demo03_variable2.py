import tensorflow as tf  
state = tf.Variable(0)  
one = tf.constant(1)  
#计算当前需要更新的值  
new_value = tf.add(state,one)  
#创建一个op完成更新操作，属性与动作分离开来  
update = tf.assign(state,new_value)  
  
init_op = tf.initialize_all_variables()  
with tf.Session() as sess:  
    sess.run(init_op)  
    print(sess.run(state))  
    for _ in range(10):  
        sess.run(update)  
        print(sess.run(state)) 