# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 11:34 AM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

import matplotlib.pyplot as plt

# 使用num生成200个随机点
# 从(-0.5,0.5)均匀的生成200个点
# x_data =np.linspace(-0.5,0.5,200)
# 上面的数据增加维度
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]

noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构建简答的神经网络的中间层
weight_mid = tf.Variable(tf.random_normal([1, 10]))
biase_mid = tf.Variable(tf.zeros([1, 10]))
result_mid = tf.matmul(x, weight_mid) + biase_mid

# 定义输出层
weight_out = tf.Variable(tf.random_normal([10, 1]))
biase_out = tf.Variable(tf.zeros([1, 1]))

result_out = tf.matmul(result_mid, weight_out) + biase_out

prediction = tf.nn.tanh(result_out)

# 二次代价函数

loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降
train_tep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as session:
    # 变量的初始化
    session.run(tf.global_variables_initializer())
    for _ in range(100):
        session.run(train_tep, feed_dict={
            x: x_data,
            y: y_data
        })

        # 查看训练的结果
    prediction_value = session.run(prediction, feed_dict={
        x: x_data
    })  # 画图
    plt.figure()
    plt.scatter(x_data, y_data)

    plt.plot(x_data, prediction_value,'r-',lw=5)
    plt.show()
