# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 16:34
# project   MachineLearning

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#              计算密集型           IO密集型
# 框架          tensorFlower        django,scrapy

                # cpu里面计算            http请求，磁盘操作


# 图默认已经注册，一组表示tf.Option计算的操作
a = tf.constant(5.0)
b = tf.constant(6.0)
sum1 = tf.add(a,b)

# 默认的这张图，相当于是给成程序分配一段内存
graph = tf.get_default_graph()
print(graph)

with tf.Session() as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)



print("*"*100)
# 创建一张图包含一组ophe tensor,上下文环境

# op:只要使用tensorflower的API定义的函数都是OP
# 比如a=1这个是python的定义形式，就不是了
# TensorFlower声明的才是
# tensor：就指代的是数据


g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)