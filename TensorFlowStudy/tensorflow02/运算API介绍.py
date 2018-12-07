# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 19:29
# project   MachineLearning


# 转换静态形状的时候
# 一个类型的N维度的数组(tf.Tensor)

# 固定值的张量

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a = tf.constant(12)
#
# zero = tf.zeros([3, 4], tf.float32)
# print(zero.eval())
#
# # 正态分布(高斯分布)
#
# tf.cast([[1,2,3],[4,5,6]],tf.float32)
# 正态分布的 4X4X4 三维矩阵，平均值 0， 标准差 1
normal = tf.truncated_normal([4, 4, 4], mean=0.0, stddev=1.0)

a = tf.Variable(tf.random_normal([2,2],seed=1))
b = tf.Variable(tf.truncated_normal([2,2],seed=2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))



print("*"*100)
# 该函数作用与squeeze相反,添加一个指定维度
sess = tf.Session()
data = tf.constant([[1, 2, 1], [3, 1, 1]])
print (sess.run(tf.shape(data)))
d_1 = tf.expand_dims(data, 0)
print (sess.run(tf.shape(d_1)))
# 数字代表下标
d_1 = tf.expand_dims(d_1, 2)
print( sess.run(tf.shape(d_1)))
d_1 = tf.expand_dims(d_1, -1)
print (sess.run(tf.shape(d_1)))

print("*"*100)
# 数字均是下标哦
sess = tf.Session()
data = tf.constant([[1, 2, 1], [3, 1, 1]])
print (sess.run(tf.shape(data)))
d_1 = tf.expand_dims(data, 0)
d_1 = tf.expand_dims(d_1, 2)
d_1 = tf.expand_dims(d_1, -1)
d_1 = tf.expand_dims(d_1, -1)
print (sess.run(tf.shape(d_1)))
d_2 = d_1
print (sess.run(tf.shape(tf.squeeze(d_1))))
print (sess.run(tf.shape(tf.squeeze(d_2, [2, 4]))))