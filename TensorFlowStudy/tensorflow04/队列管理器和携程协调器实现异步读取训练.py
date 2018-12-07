# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午3:00
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 队列管理器

# 模拟异步子线程 存入样本 主线程  读取样本
#
# 1. 定义一个而队列，10000
Q = tf.FIFOQueue(1000, tf.float32)

print("*" * 100)
# 2. 定义子线程要做的事情， 循环  值  +1 放入队列当中
var = tf.Variable(0.0)  # 变量op

# data  =var + 1              # 加法op,所以时间是不能鞥进行假发运算的
# 实现一个自增，tf.assign_add  # 这个也不是变量op类型
# 这个data和var还不是一个东西, 但是assingn_add把var本身的值变成了1.0，下次自增变2.0
data = tf.assign_add(var, tf.constant(1.0))

en_q = Q.enqueue(data)

print("*" * 100)
# 3. 定义队列管理器op, 指定子线程该干什么事情 ， 多少个子线程该干什么事情,

qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 4)  # 返回队列管理器队列

# 初始化变量OP
init_op = tf.global_variables_initializer()



with tf.Session()  as session:
    # 初始化变量
    session.run(init_op)

    # 真正开启子线程
    threads = qr.create_threads(session, start=True)

    # 主线程不断读取数据训练
    for i in range(300):
        print(session.run(Q.dequeue()))



# 程序运行完Session关闭，资源释放，子线程不能操作了，所有程序会报错
# 所以需要回收子线程