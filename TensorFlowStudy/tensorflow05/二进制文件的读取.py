# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午8:33
# project   MachineLearning

# 60000  32*32*3=3072   二进制

# CIFAR-10二进制文件数据读取
import os
import tensorflow as tf

# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("./CIFAR/cifar-10-batches-bin/")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/data.bin")


class CifarRead(object):
    """
    完成读取二进制文件，写进tfrecords，读取tfrecords
    """

    def __init__(self, filelist):
        # 文件列表
        self.filelist = filelist
        # 定义读取图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制文件每张图片的字节
        self.label_bytes = 1  # 标签
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_decode(self):
        # 1. 构造问价队列、
        file_queue = tf.train.string_input_producer(self.filelist)

        # 2. 构造二进制文件读取器，读取内容
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read((file_queue))  # 读取的是一个数据
        print(value)

        # 3. 解码内容,二进制文件解码
        label_image = tf.decode_raw(value, tf.uint8)

        print(label_image)

        # 4. 分割出图片和标签数据，切出特征值和目标值

        # 取出来是字符串，需要转成成int类型的数据
        # label = tf.slice(label_image, [0], [self.bytes])
        label = tf.cast(tf.slice(label_image, [0], [self.bytes]), tf.int32)

        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        print(label, image)

        # 5. 可以对图片的特征数据进行形状的改变[3072]-->[32,32,32]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        print(label, image_reshape)

        # 6. 批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=10, capacity=12)

        print(image_batch, label_batch)
        return image_batch, label_batch


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    cf = CifarRead(filelist)

    image_batch, label_batch = cf.read_decode()

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)

        print(session.run([image_batch, label_batch]))
        coord.request_stop()
        coord.join()
