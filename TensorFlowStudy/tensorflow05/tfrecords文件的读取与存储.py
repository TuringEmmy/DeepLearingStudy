# life is short, you need use python to create something!
# author    TuringEmmy
# time      18-11-22 下午9:45
# project   MachineLearning

# tensorflow自带文件格式
# 1.方便读取和移动
#
# 样本 = 特征值 + 目标值

# 对于每一个样本，都要构造哦example协议块
import tensorflow as tf
import os

# example=tf.train.Example(features=tf.train.Features(feature={
#     "image":
#     "label":
# }))


# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("./CIFAR/")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/data.bin", "filedir")
tf.app.flags.DEFINE_string("cifar_dir", "./CIFAR/temp.tfrecords", "save to tfrecords file")


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


if __name__ == '__main__':
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    cf = CifarRead(filelist)

    image_batch, label_batch = cf.read_decode()

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord=coord)

        print("*"*100)
        print('开始存储')
        # 因为有个eval所以 必须写到Session当中
        cf.write_tfrecords(image_batch,label_batch)
        print('结束存储')

        print(session.run([image_batch, label_batch]))
        coord.request_stop()
        coord.join()
