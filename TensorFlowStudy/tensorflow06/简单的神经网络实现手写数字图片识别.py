# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/26/18 2:52 PM
# project   DeepLearingStudy


import tensorflow as tf

# 通过api获取数据
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/', one_hot=True)
# print(mnist)

print(mnist.train.images)
print(mnist.train.labels)

# 批次获取多个数据
print(mnist.train.next_batch(50))


# 单层(全连接层)实现手写数字识别
#         特征值[None,784]                 目标值[None,10]
# 1. 定义数据占位符
# 特征值【None,784]    目标值[None,10]
# 2. 建立模型
# 随机初始化权重和偏重
# 3. 计算损失
# loss平均样本损失
# 4. 梯度下降优化