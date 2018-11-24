# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/22 13:46
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 计算：CPU操作计算
# tensorflow
# 多线程 并行的执行任务

# 队列 文件的改善（tfrecords）

# 模拟一下同步先处理数据，然后才能取数据训练

# tensorflow当中，运行操作有依赖性

# 1. 首先定义队列
Q = tf.FIFOQueue(3, tf.float32)

# 放入一些数据
enq_manny = Q.enqueue_many([[0.1, 0.2, 0.3],])
# 2. 定义一些处理数据，取数据的过程，+1,再入队列
out_q = Q.dequeue()  # 这个数据是op

data = out_q + 1

en_q = Q.enqueue(data)

with tf.Session()  as session:
    # 初始化队列
    session.run(enq_manny)

    # 模拟处理数据
    for i in range(100):
        session.run(en_q)

    # 训练数据
    for i in range(Q.size().eval()):  # Q.=ize是一个op哦，所哟的使用eval
        print(session.run(Q.dequeue()))

    pass
