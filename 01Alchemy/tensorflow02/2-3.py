# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 10:50 AM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)

mul = tf.multiply(input1, add)

with tf.Session() as sess:
    print(sess.run([mul, add]))
    print(sess.run(mul))





# =======================================================
# Feed
# 创建占位符,之后再c传入值
number1 = tf.placeholder(tf.float32)
number2 = tf.placeholder(tf.float32)

output = tf.multiply(number1, number2)

with tf.Session() as session:
    # feed的数据以字典的形式传入
    print(session.run(output, feed_dict={
        number1: [8],
        number2: [2]
    }))
