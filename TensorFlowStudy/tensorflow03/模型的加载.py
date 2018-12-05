# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/22 11:31
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear_regression():
    with tf.variable_scope("data"):
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("models"):
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
        bias = tf.Variable(0.0, name='b')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init_op = tf.global_variables_initializer()

    print("*" * 100)

    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)

    merged = tf.summary.merge_all()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init_op)
        print('随机初始化的参数权重:%f, 偏置为%f' % (weight.eval(), bias.eval()))

        filewrite = tf.summary.FileWriter('./trainable/', graph=session.graph)
        # tensorboard --logdir="C:\Users\TuringEmmy\Desktop\MachineLearning\TensorFlowStudy\tensorflow03手写识别softmax\trainable"

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的结果开始
        if os.path.exists("./save/mmodel/checkpoint"):
            saver.restore(session,'./save/mmodel/')

        for i in range(500):
            session.run(train_op)

            summary = session.run(merged)
            filewrite.add_summary(summary, i)
            print('第%d次随机初始化的参数权重:%f, 偏置为%f' % (i, weight.eval(), bias.eval()))

        # # 模型的保存
        # saver.save(session,'./save/mmodel')


    return None


if __name__ == '__main__':
    linear_regression()
