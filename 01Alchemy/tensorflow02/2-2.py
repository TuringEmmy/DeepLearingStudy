# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 9:47 AM
# project   DeepLearingStudy

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x = tf.Variable([1, 2])

a = tf.Variable([3, 4])

# 减法
sub = tf.subtract(x, a)
# 加法
add = tf.add(x, a)

# 创建一个变量初始化为0
state = tf.Variable(0, name='counter')
# 创建一个op, 作用是使state加1
new_value = tf.add(state, 1)
# 赋值
update = tf.assign(state,new_value)


init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(sub))
    print(session.run(add))
    for i in range(10):
        print(session.run(update))





