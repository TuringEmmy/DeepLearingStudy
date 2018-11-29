# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 15:49
# project   MachineLearning

import tensorflow as tf

# 下面的代码实现警告祛除
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 实现一个加法
a = tf.constant(5.0)
b = tf.constant(6.0)

# print(a, b)

sum1 = tf.add(a,b)
# print(sum1)


with tf.Session() as sess:
    print(sess.run(sum1))