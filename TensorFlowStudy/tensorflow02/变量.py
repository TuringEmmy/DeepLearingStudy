# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 20:53
# project   MachineLearning

# 变量op
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1, 2, 3, 4, 5])

var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0))
print(a, var)

# 必须做一步显示的初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)
    print(sess.run([a, var]))

print("*"*100)
# 1. 变量能够保存，普通的张量op是不行的
# 2. 当定义一个变量op的时候，一定要在会话当中去运行
# 3. name参数：在tensorboard使用的时候显示名字
