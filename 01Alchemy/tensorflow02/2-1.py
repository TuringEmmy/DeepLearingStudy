# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 9:34 AM
# project   DeepLearingStudy

import tensorflow as tf
# 下面的代码实现警告祛除
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一个常量

m1 = tf.constant([[3, 3]])

m2 = tf.constant([[2],
                  [2]])

# 创建一个矩阵乘法矩阵

product = tf.matmul(m1, m2)

# print(product)

# inint =tf.global_variables_initializer()
with tf.Session() as session:
    # session.run(inint)
    print(session.run(product))
