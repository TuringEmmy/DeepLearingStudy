# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 16:53
# project   MachineLearning

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#
# TensorFlower:前端系统---->> 定义程序的图的结构
#             后端系统---->> 运算图的结构

# 1. 运行图的结构
# 2. 分配资源的计算
# 3. 掌握资源（变量的资源，队列，线程）

# 会话相当于一个中管，一一旦关闭，里面的资源就不English在及性能那个是用了

a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum1))