# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/3/18 1:48 PM
# project   DeepLearingStudy

# 使用tensorflow来实现手写识别

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

# 载入训练集
mnist = input_data.read_data_sets('/mnt/hgfs/WorkSpace/data/MNIST', one_hot=True)

# 每批次的大小

batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
weight = tf.Variable(tf.zeros([784, 10]))
biase = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, weight) + biase)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 大小一样,返回True
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 将bool类型的转换为浮点型,然后转换为float类型,使1,0类型的
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    session.run(init)

    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            session.run(train_step, feed_dict={
                x: batch_xs,
                y: batch_ys
            })
        acc = session.run(accuracy, feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels
        })
        print("Iter" + str(epoch), ", Testing Accuracy" + str(acc))
        # print('Iter:{},Testing Accuracy{]'.format(str(epoch), str(acc)))
