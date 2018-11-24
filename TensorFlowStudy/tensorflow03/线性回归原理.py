# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 21:48
# project   MachineLearning
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 准备好1特征和1目标值 100[100,1]
#             y=x*0.7 + 0.8
# 2. 建立模型 随机初始化准备一个权重w,一个偏置信b
#         y_predict = x*w + b(模型的参数必须用该变量定义)
# 3. 求损失函数，误差
# loss误差   (y_1-y_1')^2+...+(y_100-y_100')^2   /  100
# 4. 梯度下降去优化损失过程  指定学习率
#
# 矩阵相乘：(m行, n行)  (n行, 1)  (m行, 1)   +    bias
#
# # 均镇运算：
# tf.matmul(x,w)
# # 平方：
# tf.square(error)
# # 梯度下降：
# tf.train.AdagradOptimizer(learning_rate=0.3)