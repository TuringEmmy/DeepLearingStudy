# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午9:24
# project   MachineLearning

import tensorflow as tf

a = [9, 23, 45, 29, 10]

result = tf.slice(a, [0], [4]).eval()
print(result)
