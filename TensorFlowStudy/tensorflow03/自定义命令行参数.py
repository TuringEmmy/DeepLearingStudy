# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/22 11:48
# project   MachineLearning

import tensorflow as tf

# 定有命令行参数
# 1. 首先定义有哪些参数需要在运行时候指定
# 2. 程序当中获取定义命令行参数

tf.app.flags.DEFINE_interger("max_step", 100, '模型训练的步数')
