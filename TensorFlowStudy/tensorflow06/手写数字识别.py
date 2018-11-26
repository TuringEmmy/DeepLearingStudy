# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/26/18 3:37 PM
# project   DeepLearingStudy

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def fullconnected():
    # 获取真实的数据
    mnist = input_data.read_data_sets('./data/', one_hot=True)

    # 1. 建立数据的占位符  x   [None, 784]   y_true   [None, 10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 建立一个全连接层的神将网络  w [784,10]   b [10]
    with tf.variable_scope('fc_model'):
        # 随机初始化权重和偏执
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='w')

        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        # 预测None个样本的输出结果[None,784]*[784,10]+[10]=[None,10]
        y_predict = tf.matmul(x, weight) + bias

    # 3. 求出所有样本的损失，然后求平均值
    with tf.variable_scope('soft_cross'):
        # 求平均交叉熵损失
        # 返回列表的所有值的平均值
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 梯度下降求出损失
    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

    # 5. 计算准确率
    with tf.variable_scope('acc'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list   none个样本  [1,0,1,0,0,1,1，。。。。。。]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    print("*" * 100)
    # 收集变量-单个数字
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)
    # 收集数据-高维度变量
    # 注意：高维度所用的函数
    tf.summary.histogram('weights', weight)
    tf.summary.histogram("biases", bias)
    # 合并一个变量的op
    merged = tf.summary.merge_all()
    print("*" * 100)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()
    # 开启回话，去训练
    with tf.Session() as session:
        # 初始化变量
        session.run(init_op)

        # 建立events文件
        print("*" * 100)
        filtewriter = tf.summary.FileWriter("./summary/", graph=session.graph)
        print("*" * 100)

        # 迭代步数去训练, 更新参数预测
        for i in range(200):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            # 运行train_op
            session.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # 写入每步训练的值
            print("*" * 100)
            summary = session.run(merged,feed_dict={x: mnist_x, y_true: mnist_y})
            filtewriter.add_summary(summary,i)

            print("*" * 100)

            # feed_dict必须传入，因为需要实施变化，danza这里美誉用，但又必须传入
            print("训练%d步, 准确率为：%f" % (i, session.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
    return None


if __name__ == '__main__':
    fullconnected()


"""
tensorboard --logdir="./summary/"
"""