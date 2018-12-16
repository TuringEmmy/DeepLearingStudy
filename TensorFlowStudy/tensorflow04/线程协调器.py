# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午3:30
# project   MachineLearning

# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午3:00
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 线程协调器

Q = tf.FIFOQueue(1000, tf.float32)

var = tf.Variable(0.0)

data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue(data)

qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 4)

init_op = tf.global_variables_initializer()

with tf.Session()  as session:
    session.run(init_op)

    # 开启线程管理器
    coord = tf.train.Coordinator()  # 这个是主线程开启
    threads = qr.create_threads(session, coord=coord, start=True)

    for i in range(300):
        print(session.run(Q.dequeue()))

    # 回收主线程
    coord.request_stop()
    coord.join(threads)
