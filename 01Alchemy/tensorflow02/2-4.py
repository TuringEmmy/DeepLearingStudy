# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 10:59 AM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

# 使用numpy随机生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(1)
k = tf.Variable(2)
y = k * x_data + b

# 二次代价函数,也就是所说的方差
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个帝都下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 最小化代价函数
train = optimizer.minimize(loss)
# loss越小,越接近真实值

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for step in range(200):
        session.run(train)
        if step % 20 == 0:
            print(step, session.run([k, b]))
