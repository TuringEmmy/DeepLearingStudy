# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/22 11:16
# project   MachineLearning


import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 添加权重参数，损失值扥在tensorFlow官产情况

# 1. 收集参数   注意是卸载会话之前
# 学习率和步数的设置
# 2. 合并擦变量写入事件文件

def linear_regression():
    with tf.variable_scope("data"):
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
        bias = tf.Variable(0.0, name='b')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init_op = tf.global_variables_initializer()

    print("*" * 100)
    # 收集tensor
    tf.summary.scalar("losses", loss)  # losses是在后台进行显示的
    tf.summary.histogram("weights", weight)

    # 定义合并TensorFlow的op
    merged = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(init_op)
        print('随机初始化的参数权重:%f, 偏置为%f' % (weight.eval(), bias.eval()))

        # 建立事件文件
        filewrite = tf.summary.FileWriter('./trainable/', graph=session.graph)
        # tensorboard --logdir="C:\Users\TuringEmmy\Desktop\MachineLearning\TensorFlowStudy\tensorflow03\trainable"

        for i in range(500):
            session.run(train_op)
            # 运行合并的tensor
            summary = session.run(merged)
            filewrite.add_summary(summary, i)
            print('第%d次随机初始化的参数权重:%f, 偏置为%f' % (i, weight.eval(), bias.eval()))
    return None


if __name__ == '__main__':
    linear_regression()
