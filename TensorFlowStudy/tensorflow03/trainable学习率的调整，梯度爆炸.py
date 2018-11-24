# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 23:03
# project   MachineLearning
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def linear_regression():

    # 在这里建立一个作用域
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

    with tf.Session() as session:
        session.run(init_op)
        print('随机初始化的参数权重:%f, 偏置为%f'%(weight.eval(), bias.eval()))

        # 建立事件文件
        filewrite = tf.summary.FileWriter('./trainable/',graph=session.graph)
        # tensorboard --logdir="C:\Users\TuringEmmy\Desktop\MachineLearning\TensorFlowStudy\tensorflow03\trainable"

        for i in range(500):
            session.run(train_op)
            print('第%d次随机初始化的参数权重:%f, 偏置为%f' % (i, weight.eval(), bias.eval()))
    return None



if __name__ == '__main__':
    linear_regression()


"""
添加作用域的好处就是在图像当中更加清晰的观看，也方便在代码当中进行观看
"""