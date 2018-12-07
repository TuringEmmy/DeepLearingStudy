# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 4:41 PM
# project   DeepLearingStudy
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/mnt/hgfs/WorkSpace/data/MNIST/", one_hot=True)
print("Download Done!")

x = tf.placeholder(tf.float32, [None, 784])

# paras
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

# loss func
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()
