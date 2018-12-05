# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午10:25
# project   MachineLearning


import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("./CIFAR/cifar-10-batches-bin/")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/data.bin", "filedir")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/temp.tfrecords", "save to tfrecords file")


class CifarRead(object):

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

    def write_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和目标值存进tfrecords当中
        :param image_batch: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        # 1. 构造一个tfrecords文件的存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 2. 循环的讲所有的样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i个图片数据的特征值和目标值
            # image_batch[i]这是一个类型，一定要eval,不然没有值
            image = image_batch[i].eval().tostring()

            # label=label_batch[i].eval()# 这是一个张量
            label = label_batch[i].eval()[0]  # 这是一个张量

            # 构造一个样本的example
            example = tf.train.Example(features=tf.train.Features(feature={
                "iamge": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        # 关闭
        writer.close()
        return None

    def read_tfrecords(self):
        # 1. 构造文件阅读器
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2. 构造文件阅读器，读取内容example,是一个样本的序列化example
        reader = tf.TFRecordReader()
        key, value = reader.read((file_queue))

        # 3. 解析example
        tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        print(feature["image", feature["label"]])
        # 4. 解码内容,如果读取的内容格式是string需要解码， 如果是int64
        image = tf.decode_raw(features["image"], tf.uint8)

        # 固定图片的形状
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        label = tf.cast(features["label"], tf.int32)

        print(image, label)

        # 进行批处理
        image_batch,label_batch=tf.train.batch([image_reshape, label],batch_size=10,num_threads=3,capacity=10)
        return image_batch,label_batch


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    cf = CifarRead(filelist)

    image_batch, label_batch = cf.read_tfrecords()

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)

        print(session.run(image_batch,label_batch))
        coord.request_stop()
        coord.join()
