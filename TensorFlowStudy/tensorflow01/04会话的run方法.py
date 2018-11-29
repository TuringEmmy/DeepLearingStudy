# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 17:11
# project   MachineLearning


import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)

with tf.Session() as sess:
    # run(self, fetches, feed_dict=None, options=None, run_metadata=None)
    print(sess.run(sum1))

print("*"*100)
# 不是tensor类型的不能运行
var1 =2
# var2 =3
# sum2=var1+var2
# with tf.Session() as sess:
#     print(sess.run(sum2))
# 但是可以重载
sum2 = a+var1
print(sum2)
with tf.Session() as sess:
    print(sess.run(sum2))



print("*"*100)
# 训练模型
# 实时的提供数据进行训练

# placehoder是一个占位符
# placeholder(dtype, shape=None, name=None)
# 下面这个也是op哦,feed_dict是一个字典哦
plt = tf.placeholder(tf.float32,[2,3])

with tf.Session() as sess:
    print(sess.run(plt,feed_dict={plt:
                                  [[1,2,3],
                                   [4,5,6]
                                   ]}))
print("*"*100)
plt = tf.placeholder(tf.float32,[None,3])
# None的作用可以自定义下面的输入
with tf.Session() as sess:
    print(sess.run(plt,feed_dict={plt:
                                  [[1,2,3],
                                   [4,5,6],
                                   [7,8,9],
                                   [12,13,34]
                                   ]}))