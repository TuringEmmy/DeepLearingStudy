# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/29/18 10:10 AM
# project   DeepLearingStudy
# 识别数字如OCR

# googlnet



# 了解常用API
# [None, 784]  [None,10]
# 卷积神经网络：
# 卷基层：卷积   32个filter,5*5  strides 1, padding ='SAME'  same说明凸显输入大小是一样的
# 输入：[None, 28,28,1]  输出：[None, 28,28,32]
#     激活: [None, 28, 28, 32]
#     池化： 2*2 ，strides2, padding='SAME
#             [None, 28, 28, 32]  ------>  [None, 14 ,14 32]



# 二卷积层：　卷积64个filter, 5*5, strides1, padding='SAME
# 输入：[None,14,14,32]  输出：[None,14,14,64]
#     激活: [None, 14,14 ,64]
#     池化:2*2 strids2
#                 [None,14,14,64]   ------>  [None,7,7,64]


# 全连接层FC:
#         [None,7*7*64]-->[7*7*64,10]-->[None,10]





from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    bias = tf.Variable(tf.constant(0.0, shape=shape))
    return bias


def model():
    """
    自定义的卷积模型
    :return:
    """
    # 1. 准备数据的占位符  x [None, 784] y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 一卷积层  卷积5×5*1  32  strides=1 激活  池化
    with tf.variable_scope('conv1'):
        # 随机初始化权重, 偏置【32】
        w_conv1 = weight_variables([5, 5, 1, 32])

        b_conv1 = bias_variables([32])

        # 对x进行形状的改变[None,784]  [None,28,28,1]
        # 对不知道的-1代替
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 从一个[None,28,28,1]----->[None,28,28,32]
        ret1 = tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1

        # 激活
        x_relu1 = tf.nn.relu((ret1))
        # 池化2*2  strides2 [None,28,28,32]-->[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 1, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 3. 二卷积层
    with tf.variable_scope("conv2"):
        # 5*5每个人带32个5*5 共64个人
        w_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variables([64])

        # 卷积，激活，池化
        ret2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

        # 激活 [None.14.14.32]-->[None,14,14,64]
        x_relu2 = tf.nn.relu(ret2)
        # 池化 2*2 strides 2  [None,14,14,64]--->[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 4.全链接层[None,7,7,64]--->[None,7*7*64]*[7*&*64,10]+[10] =[None,10]
    with tf.variable_scope("fc"):
        # 输出每个样本的10个样本的值
        # 随机初始化权重和偏置
        w_fc = weight_variables([7 * 7 * 64, 10])
        b_fc = bias_variables([10])

        # 修改形状[None,7,7,64]---> [None,7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])  # 特征值

        # 进行矩阵运算得出每个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    # 获取真实的数据
    mnist = input_data.read_data_sets("./data/", one_hot=True)

    x, y_true, y_predict = model()

    # 进行交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降求损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 计算精确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化的op
    init_op = tf.global_variables_initializer()

    # 开启回话运行
    with tf.Session() as session:
        session.run(init_op)
        # 循环去训练
        for i in range(1000):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            # 运行train_op训练
            session.run(train_op, feed_dict={
                x: mnist_x,  # 特征值
                y_true: mnist_y  # 目标值
            })
            print("训练第{},准确率为：{}".format(i,
                                         session.run(accuracy, feed_dict={
                                             x: mnist_x,
                                             y_true: mnist_y
                                         })))
    return None


if __name__ == '__main__':
    conv_fc()
