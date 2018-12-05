# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午3:40
# project   MachineLearning

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 比如csv文件
#
# 1. 构造一个文件队列[将文件的路径+名字]加入队列
# 2. 构造文件阅读器读取对垒内容,解码
#         read{这里要分文件格式了
#              (cv文件一行):读取一行
#             (二进制文件) :指定一个样本的bytes读取
#             (图片文件)：按一张一张的读取}
#         默认读取：默认值读取一个样本
# 3. 读取队列内容
# 结果：一个样本的内容
# 4. 批处理


print("*"*100)
# 主线程要做：取样本数据训练
#
# A文件
#
# B文件
#
# C文件
#
# D文件
#
# 100个样本