# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 17:55
# project   MachineLearning

import tensorflow as tf

# 下面的代码实现警告祛除
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)
# plt = tf.placeholder(tf.float32, [None, 3])
plt = tf.placeholder(tf.float32, [2, 3, 6])

with tf.Session() as sess:
    # 打印a的形状
    print(a.shape)
    print(a.graph)
    print(plt.shape)
    print(a.op)

# 在TensorFlow当中：打印出来的形状表示
# 0维：()
# 1维:(5)
# 2维：(5,6)  , (?,10)
# 3维：(2,3,5)

print("*" * 100)
# 形状方便面的一些常识、


# Numpy:resahpe把原来的数据通过知己而修改
# tensorflowe
# 动态形状和静态形状
#     动态：
# 静态形状
plt = tf.placeholder(tf.float32, [None, 2])
print(plt)
# 固定形状去设置数据
plt.set_shape([3, 2])
print(plt)

# 注意：对于静形状来说，一旦张量形状固定，不能再次设置静态形状,跨纬度也不能
# 动态形状可以去创建一个新的张量
# plt.set_shape([4, 2])
# print(plt)
print("*" * 100)

# 这里的reshape和numpy的是不一样的
# 这里使用了动态修改
plt_reshape = tf.reshape(plt, [2, 3, 1])
print(plt_reshape)
# 但是plt_reshape = tf.reshape(plt, [3, 3])就错了，改变的时候一定药注意里面的元素数量


with tf.Session() as session:
    pass
