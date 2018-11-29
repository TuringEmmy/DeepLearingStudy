# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 20:47
# project   MachineLearning
import tensorflow as tf
import tensorboard

# tensorboard --logdir="./summary/"

# tensorboard --logdir="C:\Users\TuringEmmy\Desktop\MachineLearning\TensorFlowStudy\tensorflow03\summary"


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(3, name="a")
b=tf.constant(2,name="b")
c=tf.add(a,b,name='add')

var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0),name='variable')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 把程序的图结构写入事件, graph:把指定的图写进事件文件当中
    filewrite = tf.summary.FileWriter("./summary/", graph=sess.graph)

    print(sess.run([c, var]))
