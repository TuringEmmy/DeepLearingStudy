# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/26/18 4:52 PM
# project   DeepLearingStudy


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 定义一个命令行参数
FLAGS = tf.app.flags.FLAGS
# is_train默认为1,表示训练
tf.app.flags.DEFINE_integer("is_train", 0, "指定程序是预测还是训练")


def fullconnected():
    mnist = input_data.read_data_sets('./data/', one_hot=True)

    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    with tf.variable_scope('fc_model'):
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name='w')
        bias = tf.Variable(tf.constant(0.0, shape=[10]))
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('soft_cross'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

    with tf.variable_scope('acc'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)
    tf.summary.histogram('weights', weight)
    tf.summary.histogram("biases", bias)

    init_op = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    print("*" * 100)
    # 创建一个saver
    saver = tf.train.Saver()
    print("*" * 100)

    with tf.Session() as session:
        session.run(init_op)
        filtewriter = tf.summary.FileWriter("./summary/", graph=session.graph)

        if FLAGS.is_train == True:
            # 训练模型
            for i in range(200):
                mnist_x, mnist_y = mnist.train.next_batch(50)
                session.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                summary = session.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                filtewriter.add_summary(summary, i)
                print("训练第%d步, 准确率为：%f" % (i, session.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

            print("*" * 100)
            # 保存模型
            saver.save(session, "./model/fc_model")
            print("*" * 100)

        else:
            # 加载模型
            saver.restore(session, './model/fc_model')
            # 如果是0, 做出预测
            for i in range(100):
                # 每次测试一张图片
                x_test, y_test = mnist.test.next_batch(1)

                print("第%d张图片, 手写目标是%d, 预测结果是:%d" % (
                    i,
                    tf.argmax(y_test, 1).eval(),
                    # 返回这张图片的属于每个类别的概率
                    # session.run(y_predict,feed_dict={x: x_test, y_true: y_test})
                    tf.argmax(session.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                ))

    return None


if __name__ == '__main__':
    fullconnected()

"""
tensorboard --logdir="./summary/"
"""
