# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 22:16
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear_regression():
    """
    自实现一个线性回归预测
    :return:
    """
    # 1. 准备数据， x 特征值  [100, 10]       y 目标值   [100]
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')

    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 2. 建立线性回归模型   1 个特征  1个权重         1偏置 y = wx+b
    # 随机给一个权重和偏置的值，让它去计算你损失，然后在当前状态下优化
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
    # 用变量定义才能优化
    bias = tf.Variable(0.0, name='b')

    y_predict = tf.matmul(x, weight) + bias

    # 3. 建立损失函数, 均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4. 优化损失<梯度下降优化损失> 0~1   2，  ，3   4，
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 通过会话运行程序
    with tf.Session() as session:
        # 初始化变量
        session.run(init_op)

        # 打印随机最先初始化的权重和偏置
        print('随机初始化的参数权重:%f', (weight.eval(), bias.eval()))

        # 循环运行优化
        for i in range(500):
            session.run(train_op)
            print('第%f次随机初始化的参数权重:%f, 偏置为%f'% (i, weight.eval(), bias.eval()))

    return None


if __name__ == '__main__':
    linear_regression()
    pass
