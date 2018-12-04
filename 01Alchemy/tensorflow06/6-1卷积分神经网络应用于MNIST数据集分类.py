# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/4/18 12:53 PM
# project   DeepLearingStudy

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/mnt/hgfs/WorkSpace/data/MNIST", one_hot=True)
batch_size = 100

n_batch = mnist.train.num_examples // batch_size


# 初始化权重
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化权重
def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# ================================卷积层1===============================================
# 初始化第一个卷基层权重和偏置
w_conv1 = weight_variables([5, 5, 1, 32])  # 5*5的采样窗口,32个卷积从1个平面抽取特征
b_conv1 = bias_variables([32])

# 第一个卷积层
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# 进行max_pooling
h_pool1 = max_pool_2x2(h_conv1)

# ================================卷积层2============================================
# 初始化第一个卷基层权重和偏置
w_conv2 = weight_variables([5, 5, 32, 64])  # 5×5的采样窗口，64个卷积从32个平面抽取特征
b_conv2 = bias_variables([64])

# 第二个卷基层
# relu进行激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 28×28的图片第一次卷积后还有28×28,第一次池化为14×14
# 第二次卷积后为14×14,第二次池化后变为7×7
# 上面的操作之后，变成64张7×7的平面



# ===========================全连接层1========================================
# 上一层有7×7×64个神经元，全连接层有1024个神经元
w_fc1 = weight_variables([7 * 7 * 64, 1024])
b_fc1 = bias_variables([1024])  # 1024个节点

# 将池化层2的输出扁平化成1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


# keep_prob表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


# 初始化第二个全了连接层
w_fc2=weight_variables([1024,10])
b_fc2=bias_variables([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

correct_prediction =tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))        # 结果存入布尔列表当中

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={
                x:batch_xs,
                y:batch_ys
            })
        acc = sess.run(accuracy,feed_dict={
            x:mnist.test.images,
            y:mnist.test.labels,
            keep_prob:1.0
        })

        print("计算第{}"+str(epoch)+"步, 计算准确率为"+str(acc)+"!")
