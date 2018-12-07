# life is short, you need use python to create something!
# author    TuringEmmy
# time      11/29/18 3:25 PM
# project   DeepLearingStudy

import os
print(os.listdir("./"))

import tensorflow as tf
with tf.Session() as sess:
    ret=tf.one_hot([[13, 45, 18, 18], [19, 16, 15, 89]], depth=26, axis=2, on_value=1.0).eval()
    # depth编码之后的数量
    # on_value是就写1.0
    print(ret)


